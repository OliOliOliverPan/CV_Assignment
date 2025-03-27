from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg'))])
        assert len(self.image_filenames) == len(self.mask_filenames), "Image/mask count mismatch"

        self.img_transform = transform
        self.mask_transform = transforms.ToTensor()  # We'll do additional logic after this

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]
    
        # Load image and mask
        image_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
    
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)  # Don't convert to L, let ToTensor handle it
    
        # Apply image transform
        if self.img_transform:
            image = self.img_transform(image)
    
        # Convert mask to tensor (shape: [1, H, W] → [H, W])
        mask_tensor = self.mask_transform(mask).squeeze(0)
    
        # Initialize output class mask
        class_mask = torch.zeros_like(mask_tensor, dtype=torch.long)
    
        # Rule 1: background = black = 0.0
        is_background = mask_tensor == 0.0
    
        # Rule 2: boundary = white = 1.0
        is_boundary = mask_tensor == 1.0
    
        # Rule 3: animal = in between
        is_animal = (~is_background) & (~is_boundary)
    
        # Apply class labels
        class_mask[is_background] = 2
        class_mask[is_boundary] = 3
    
        # Determine cat (0) or dog (1) by **mask_name**
        if mask_name[0].isupper():
            class_mask[is_animal] = 0  # Cat
        else:
            class_mask[is_animal] = 1  # Dog
    
        return image, class_mask, os.path.splitext(img_name)[0], os.path.splitext(mask_name)[0]