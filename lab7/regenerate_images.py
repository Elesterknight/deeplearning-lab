import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
image_size = 64
nc = 3
nz = 100
ngf = 64
batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights_path = './model_weight/Generator_weights.pth'

# --- 1. Regenerate Real Training Images Plot ---
print("Loading dataset...")
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Use 'test' split as in train.py or just a subset
dataset = dsets.Flowers102(root='./data', split='test', transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

real_batch = next(iter(dataloader))

plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
# Correct order: imshow THEN savefig
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.savefig('real_training_images.png')
plt.close()
print("Saved real_training_images.png")

# --- 2. Regenerate Generated Images Plot ---
print("Loading Generator...")

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is Z (latent vector)
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
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

netG = Generator(1).to(device)

# Load weights (handle DataParallel saving if needed)
if os.path.exists(weights_path):
    state_dict = torch.load(weights_path, map_location=device)
    # If weights were saved from DataParallel, they have 'module.' prefix
    # Creating a new dict without 'module.' prefix if it exists
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") # remove `module.`
        new_state_dict[name] = v
    
    netG.load_state_dict(new_state_dict)
    print("Generator weights loaded.")
else:
    print(f"Error: Weights not found at {weights_path}")
    exit()

# Generate images
noise = torch.randn(64, nz, 1, 1, device=device)
with torch.no_grad():
    fake = netG(noise).detach().cpu()

plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1,2,0)))
plt.savefig('generated_final_images.png')
plt.close()
print("Saved generated_final_images.png")
