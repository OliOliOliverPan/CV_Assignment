import os
import re
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from custom_dataset import CustomDataset
from unet import UNet
if __name__ == "__main__":
    base_dir = "/home/s2103701/Dataset_filtered"

    # ====== Dataset and Dataloader ======
    train_dataset = CustomDataset(
        image_dir=os.path.join(base_dir, "train_randaugmented", "color"),
        mask_dir=os.path.join(base_dir, "train_randaugmented", "label"),
        transform=transforms.ToTensor()
    )

    def custom_collate_fn(batch):
        images, masks, img_names, mask_names = zip(*batch)

        img_names = [re.sub(r'(_aug_\d+|_orig)$', '', name) for name in img_names]
        mask_names = [re.sub(r'(_aug_\d+|_orig)$', '', name) for name in mask_names]

        images = torch.stack(images)
        return images, list(masks), list(img_names), list(mask_names)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

    val_dataset = CustomDataset(
        image_dir=os.path.join(base_dir, "val_resized", "color"),
        mask_dir=os.path.join(base_dir, "val_resized", "label"),
        transform=transforms.ToTensor()
    )
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # ====== Model & Multi-GPU ======
    model = UNet(in_channels=3, out_channels=4)

    if torch.cuda.device_count() > 1:
        print(f"🔧 Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ====== Load original sizes ======
    with open(os.path.join(base_dir, "original_sizes.json"), "r") as f:
        original_sizes = json.load(f)

    def resize_multiclass_logits(logits, img_names, original_sizes_dict):
        resized_logits = []
        for i in range(len(logits)):
            name = img_names[i]
            orig_h, orig_w = original_sizes_dict[name]
            resized = F.interpolate(
                logits[i].unsqueeze(0),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False
            )
            resized_logits.append(resized.squeeze(0))  # shape: (4, H, W)
        return resized_logits

    # ====== Training config ======
    NUM_EPOCHS = 100
    PRINT_INTERVAL = 10
    BEST_MODEL_PATH = "/home/s2103701/Model/model_100_epochs.pth"
    best_val_loss = float("inf")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    # ====== Training loop ======
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0

        for images, masks, img_names, _ in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training"):
            images = images.to(device)
            masks = [m.to(device) for m in masks]
            optimizer.zero_grad()

            logits = model(images)
            resized_logits = resize_multiclass_logits(logits, img_names, original_sizes)

            total_loss = 0
            for pred, gt in zip(resized_logits, masks):
                pred = pred.unsqueeze(0)
                gt = gt.unsqueeze(0)
                loss = loss_fn(pred, gt)
                total_loss += loss

            total_loss.backward()
            optimizer.step()
            total_train_loss += total_loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # ====== Validation ======
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for images, masks, img_names, _ in tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation"):
                images = images.to(device)
                masks = [m.to(device) for m in masks]

                logits = model(images)
                resized_logits = resize_multiclass_logits(logits, img_names, original_sizes)

                for pred, gt in zip(resized_logits, masks):
                    pred = pred.unsqueeze(0)
                    gt = gt.unsqueeze(0)
                    val_loss = loss_fn(pred, gt)
                    total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"💾 Saved new best model at epoch {epoch+1} (val loss: {avg_val_loss:.4f})")

    print("🎉 Training complete. Best validation loss:", best_val_loss)

    # ====== Final Save ======
    MODEL_PATH = Path("/home/s2103701/Model")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = "unet_100_epochs.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)