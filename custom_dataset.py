import random
import torchvision.transforms.functional as TF

class SegmentationAugment:
    def __init__(self, apply_color_jitter=True):
        self.apply_color_jitter = apply_color_jitter
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2
        )


    def __call__(self, image, mask):
        """
        Apply a single randomly chosen spatial transform to both image and mask.
        Important: Do NOT apply color changes to mask.
        """
        ops = ['flip', 'rotate', 'affine', 'none']
        op = random.choice(ops)

        if op == 'flip':
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        elif op == 'rotate':
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=255)  # 255 → will be treated as unknown

        elif op == 'affine':
            image = TF.affine(image, angle=0, translate=(10, 10), scale=1.0, shear=10, fill=0)
            mask = TF.affine(mask, angle=0, translate=(10, 10), scale=1.0, shear=10, fill=255)
        
        if self.apply_color_jitter and random.random() < 0.5:
            image = self.color_jitter(image)

        # else: do nothing

        return image, mask







import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        image_dir: Directory with input images
        mask_dir: Directory with corresponding grayscale masks
        transform: Custom transform that takes in (image, mask) and returns (aug_image, aug_mask)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        assert len(self.image_filenames) == len(self.mask_filenames), "Image/mask count mismatch"

        self.img_tensor_transform = transforms.ToTensor()
        self.mask_to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]

        image_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image, mask = self.transform(image, mask)

        image = self.img_tensor_transform(image)
        mask_tensor = self.mask_to_tensor(mask).squeeze(0)

        # Initialize final class mask
        class_mask = torch.zeros_like(mask_tensor, dtype=torch.long)
        
        # Mapping rules:
        is_background = mask_tensor == 0.0
        is_boundary = mask_tensor == 1.0
        is_catdog = ~(is_background | is_boundary)  # 其余像素

        class_mask[is_background] = 2  # Background
        class_mask[is_boundary] = 3    # Boundary

        # 根据文件名判断：若首字母大写则为 Cat，否则为 Dog
        if img_name[0].isupper():
            class_mask[is_catdog] = 0  # Cat
        else:
            class_mask[is_catdog] = 1  # Dog

        return image, class_mask, os.path.splitext(img_name)[0], os.path.splitext(mask_name)[0]