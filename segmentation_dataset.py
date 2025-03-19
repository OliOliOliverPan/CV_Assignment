import os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    """
    A custom Dataset class for image segmentation tasks.
    Expects:
      - image_dir: Directory with all the images.
      - mask_dir: Directory with all the corresponding segmentation masks.
      - transform: Optional; transformations/augmentations applied to both images & masks.
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str or Path): Path to the directory containing images.
            mask_dir (str or Path): Path to the directory containing segmentation masks.
            transform (callable, optional): A function/transform that takes in 
                                            a PIL image and mask, and returns 
                                            (transformed_image, transformed_mask).
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        # Gather all image/mask file paths (adjust file extensions as needed)
        self.image_paths = sorted(self.image_dir.glob("*.jpg"))
        self.mask_paths = sorted(self.mask_dir.glob("*.png"))

        # Basic validation check
        assert len(self.image_paths) == len(self.mask_paths), (
            "Number of images and masks do not match. "
            f"Found {len(self.image_paths)} images and {len(self.mask_paths)} masks."
        )

        self.transform = transform

    def __len__(self):
        """Return the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load and return a sample from the dataset at the given index.
        Returns:
            (image, mask): Where both are tensors if transform is applied,
                           or PIL images otherwise.
        """
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")  # Ensure 3-channel

        # Load corresponding mask
        mask_path = self.mask_paths[idx]
        # Depending on your mask format, you might not convert to RGB
        mask = Image.open(mask_path)

        # Apply transformations/augmentations (if any)
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask
    

