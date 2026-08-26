import os
import cv2
import numpy as np
import torch
import torch.utils.data as DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.config import VOC_IMG_MEAN, VOC_IMG_STD, YOLO_IMG_DIM, ANCHORS, GRID_SIZES

train_data_pipelines = A.Compose([
    A.RandomSizedBBoxSafeCrop(width=416, height=416, erosion_rate=0.2, p=0.5),
    A.Resize(height=YOLO_IMG_DIM, width=YOLO_IMG_DIM, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
    A.OneOf([
        A.Blur(blur_limit=3, p=0.5),
        A.GaussNoise(variance_limit=(10.0, 50.0), p=0.5),
    ], p=0.3),
    A.Normalize(mean=VOC_IMG_MEAN, std=VOC_IMG_STD),
    ToTensorV2(),
], bbox_params=A.BboxParams(
    format='pascal_voc',
    label_fields=['cls_labels'],
    min_visibility=0.2,
    min_area=10
))

test_data_pipelines = A.Compose([
    A.Resize(height=YOLO_IMG_DIM, width=YOLO_IMG_DIM),
    A.Normalize(mean=VOC_IMG_MEAN, std=VOC_IMG_STD),
    ToTensorV2(),
],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['cls_labels'])
)

class VocDetectorDataset(DataLoader.Dataset):
    image_size = YOLO_IMG_DIM
    def __init__(
        self,
        root_img_dir,
        dataset_file,
        train,
        contain_labels=True,
        num_classes=20,
        grid_sizes=GRID_SIZES,
        transform=None,
        return_image_id=False,
        encode_target=True,
    ):
        print("Initializing dataset")
        self.root = root_img_dir
        self.contain_labels = contain_labels
        self.train = train
        self.transform = transform if transform is not None else (train_data_pipelines if train else test_data_pipelines)
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.grid_sizes = grid_sizes
        self.num_classes = num_classes
        self.return_image_id = return_image_id
        self.encode_target = encode_target
        with open(dataset_file) as f:
            lines = f.readlines()

        for line in lines:
            split_line = line.strip().split()
            self.fnames.append(split_line[0])
            if self.contain_labels:
                num_boxes = (len(split_line) - 1) // 5
                box = []
                label = []
                for i in range(num_boxes):
                    x1 = float(split_line[1 + 5 * i])
                    y1 = float(split_line[2 + 5 * i])
                    x2 = float(split_line[3 + 5 * i])
                    y2 = float(split_line[4 + 5 * i])
                    c = split_line[5 + 5 * i]
                    box.append([x1, y1, x2, y2])
                    label.append(int(c))
                self.boxes.append(box)
                self.labels.append(label)
        self.num_samples = len(self.fnames)

    def _compute_iou_wh(self, w1, h1, w2, h2):
        intersection = torch.min(w1, w2) * torch.min(h1, h2)
        union = w1 * h1 + w2 * h2 - intersection
        return intersection / (union + 1e-16)

    def encoder(self, image, boxes:list, labels:list):
        image_height, image_width = image.shape[1], image.shape[2]
        num_scales = len(self.grid_sizes)
        
        target_tensors = [torch.zeros(gs, gs, 3, 5 + self.num_classes) for gs in self.grid_sizes]
        anchors = torch.tensor(ANCHORS, dtype=torch.float32)

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            w_pixel = x2 - x1
            h_pixel = y2 - y1
            cx_pixel = x1 + w_pixel / 2
            cy_pixel = y1 + h_pixel / 2
            
            w = w_pixel / image_width
            h = h_pixel / image_height
            
            if w <= 0 or h <= 0: continue

            iou_anchors = self._compute_iou_wh(torch.tensor(w), torch.tensor(h), anchors[..., 0], anchors[..., 1])
            best_anchor_indices = iou_anchors.argmax(dim=-1)
            
            has_anchor = [False] * num_scales
            for scale_idx, anchor_idx in enumerate(best_anchor_indices):
                grid_size = self.grid_sizes[scale_idx]
                
                grid_x = int(cx_pixel / image_width * grid_size)
                grid_y = int(cy_pixel / image_height * grid_size)
                
                if target_tensors[scale_idx][grid_y, grid_x, anchor_idx, 4] == 0:
                    target_tensors[scale_idx][grid_y, grid_x, anchor_idx, 4] = 1 # Objectness
                    
                    tx = (cx_pixel / image_width * grid_size) - grid_x
                    ty = (cy_pixel / image_height * grid_size) - grid_y
                    
                    anchor_w, anchor_h = anchors[scale_idx, anchor_idx]
                    # ############## 1.7 Bug 修正：正確編碼 tw, th ##############
                    tw = torch.log(w / anchor_w + 1e-16)
                    th = torch.log(h / anchor_h + 1e-16)
                    
                    target_tensors[scale_idx][grid_y, grid_x, anchor_idx, 0:4] = torch.tensor([tx, ty, tw, th])
                    target_tensors[scale_idx][grid_y, grid_x, anchor_idx, 5 + int(label)] = 1
                    has_anchor[scale_idx] = True

            if not any(has_anchor):
                best_iou_val = iou_anchors.max()
                best_scale_idx = iou_anchors.view(-1).argmax() // 3
                best_anchor_idx = iou_anchors.view(-1).argmax() % 3
                
                grid_size = self.grid_sizes[best_scale_idx]
                grid_x = int(cx_pixel / image_width * grid_size)
                grid_y = int(cy_pixel / image_height * grid_size)

                target_tensors[best_scale_idx][grid_y, grid_x, best_anchor_idx, 4] = 1
                tx = (cx_pixel / image_width * grid_size) - grid_x
                ty = (cy_pixel / image_height * grid_size) - grid_y
                anchor_w, anchor_h = anchors[best_scale_idx, best_anchor_idx]
                tw = torch.log(w / anchor_w + 1e-16)
                th = torch.log(h / anchor_h + 1e-16)
                target_tensors[best_scale_idx][grid_y, grid_x, best_anchor_idx, 0:4] = torch.tensor([tx, ty, tw, th])
                target_tensors[best_scale_idx][grid_y, grid_x, best_anchor_idx, 5 + int(label)] = 1

        return target_tensors

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.train:
            boxes = self.boxes[idx]
            labels = self.labels[idx]
            
            # The try-except block is a great way to handle cases where augmentation removes all boxes
            try:
                transformed = self.transform(image=img, bboxes=boxes, cls_labels=labels)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_labels = transformed['cls_labels']
                
                # If all boxes are removed, we can try again or skip
                if len(transformed_bboxes) == 0:
                    # Fallback to a simpler augmentation if the strong one fails
                    transformed = test_data_pipelines(image=img, bboxes=boxes, cls_labels=labels)
                    transformed_image, transformed_bboxes, transformed_labels = transformed['image'], transformed['bboxes'], transformed['cls_labels']

            except Exception:
                # Fallback if any other error occurs
                transformed = test_data_pipelines(image=img, bboxes=boxes, cls_labels=labels)
                transformed_image, transformed_bboxes, transformed_labels = transformed['image'], transformed['bboxes'], transformed['cls_labels']

            target = self.encoder(transformed_image, transformed_bboxes, transformed_labels)
            return transformed_image, target
        else: # For validation or testing
            boxes = self.boxes[idx] if self.contain_labels else []
            labels = self.labels[idx] if self.contain_labels else []
            transformed = self.transform(image=img, bboxes=boxes, cls_labels=labels)
            image = transformed['image']
            
            if self.encode_target:
                 target = self.encoder(image, transformed['bboxes'], transformed['cls_labels'])
                 return image, target
            else:
                 return image, transformed['bboxes'], transformed['cls_labels']

    def __len__(self):
        return self.num_samples

def collate_fn(batch):
    images = []
    if len(batch[0]) == 2: # train or val with encoded targets
        targets_list = [[] for _ in range(len(GRID_SIZES))]
        for image, target in batch:
            images.append(image)
            for scale_idx, scale_target in enumerate(target):
                targets_list[scale_idx].append(scale_target)
        
        images = torch.stack(images, dim=0)
        targets = [torch.stack(tgt_scale, dim=0) for tgt_scale in targets_list]
        return images, targets
    else: # eval, targets are not encoded
        boxes_list = []
        labels_list = []
        for image, boxes, labels in batch:
            images.append(image)
            boxes_list.append(boxes)
            labels_list.append(labels)
        images = torch.stack(images, dim=0)
        return images, boxes_list, labels_list