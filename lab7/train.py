#%matplotlib inline
import random
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim 
import torch.utils.data
import torchvision.utils as vutils
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set random seed for reproducibility
manualSeed = 0
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Number of workers for dataloader
workers = 8

# Batch size during training
batch_size = 256

# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 500

# Learning rate for optimizers
lr = 0.0001

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Number of times to update the critic before updating the generator
n_critic = 5  

# Weight clipping range
clip_value = 0.01  

from torch.utils.data import ConcatDataset

##########################################################################
# TODO: Define the data transformation pipeline.                         #
# You need to implement a Compose pipeline that includes:                #
# 1. Resizing images to the target size(64*64).                          #
# 2. Applying data augmentation techniques to increase dataset diversity #
# 3. Converting images to Tensor.                                        #
# 4. Normalizing the pixel values to the range [-1, 1].                  #
##########################################################################

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

##########################################################################
#                            End of your code                            #
##########################################################################


trainset = torchvision.datasets.Flowers102(root='./data', split='test', transform=transform, download=True)
testset = torchvision.datasets.Flowers102(root='./data', split='train', transform=transform, download=True)
validdataset = torchvision.datasets.Flowers102(root='./data', split='val', transform=transform, download=True)
dataset = ConcatDataset([trainset, testset, validdataset])

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

print("訓練集樣本數量:", len(dataset))

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.savefig('real_training_images.png')
#plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.close()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is Z (latent vector), going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf) x 32 x 32


            # Final output layer
            # State size: (ngf) x 32 x 32 -> (nc) x 64 x 64
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 16 x 16

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8) x 4 x 4


            # Final classification layer
            # State size: (ndf*8) x 4 x 4 -> 1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid() # Removed for WGAN
        )

    def forward(self, input):
        return self.main(input)

# Create the generator
netG = Generator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)

print(netG)


# Create the Discriminator
netD = Discriminator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)

print(netD)


# Fixed noise vector for observing generator's progression
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Optimizers use RMSProp, not Adam
optimizerD = optim.RMSprop(netD.parameters(), lr=lr)
optimizerG = optim.RMSprop(netG.parameters(), lr=lr)

# Training Loop
import torchvision.transforms as T
errG = torch.tensor(0.0)
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
# masked_real_cpu = add_random_mask(real_cpu.clone())
print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        
        ############################
        # (1) Update Critic (Discriminator)
        ###########################
        for p in netD.parameters():
            p.requires_grad = True

        netD.zero_grad()
        # Train with real
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        output_real = netD(real_cpu).view(-1)
        errD_real = -torch.mean(output_real)
        errD_real.backward()

        # Train with fake
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        output_fake = netD(fake.detach()).view(-1)
        errD_fake = torch.mean(output_fake)
        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizerD.step()

        # Weight clipping
        for p in netD.parameters():
            p.data.clamp_(-clip_value, clip_value)


        ############################
        # (2) Update Generator
        ###########################
        for p in netD.parameters():
            p.requires_grad = False

        netG.zero_grad()
        
        if i % n_critic == 0:
            netG.zero_grad()
            output = netD(fake).view(-1)
            errG = -torch.mean(output)
            errG.backward()
            optimizerG.step()


        # Output training status
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item()))

        # Save loss values for later plotting
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check generator's performance periodically
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1



weights_dir = './model_weight/'
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)


torch.save(netG.state_dict(), os.path.join(weights_dir, 'Generator_weights.pth'))
torch.save(netD.state_dict(), os.path.join(weights_dir, 'Discriminator_weights.pth'))

print("model weight save to 'model_weight/'")

# Generate new images and save them
noise = torch.randn(8189, nz, 1, 1, device=device)

netG.eval()
with torch.no_grad():
    fake = netG(noise)


output_dir = './GENIMG/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for j in range(fake.size(0)):
    transform = T.Compose([T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]), T.ToPILImage()])
    img = transform(fake[j].cpu())
    img.save('./GENIMG/fake' + str(j) + '.jpg')

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('loss_plot.png')
plt.close()

# Calculate FID
import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score
from PIL import Image
import os

# Resized original dataset path after 64x64 pixel adjustment
resized_folder_path = './resized_flowers102/'
# Generated image folder
generated_images_folder = './GENIMG/'
# Use Inception V3 model to calculate FID
inception_model = torchvision.models.inception_v3(pretrained=True)
fid_value = fid_score.calculate_fid_given_paths([resized_folder_path, generated_images_folder], batch_size=batch_size, device=device, dims=2048, num_workers=8)
print('FID value:', fid_value)

fig = plt.figure(figsize=(8,8))
plt.axis("off")
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Generated Images")
if len(img_list) > 0:
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.savefig('generated_final_images.png')
plt.close()

import pandas as pd

df_submission = pd.DataFrame({
    'id': [1], 
    'fid_score': [fid_value]
})

output_csv_path = 'result.csv'
df_submission.to_csv(output_csv_path, index=False)
