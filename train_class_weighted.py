import os
import re
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from custom_dataset import SegmentationAugment, CustomDataset
from unet import UNet
from enhanced_unet import EnhancedUNet

if __name__ == "__main__":
    # base_dir = "/home/s2103701/Dataset_filtered"
    base_dir = "/Users/bin/Desktop/CV_Assignment/Dataset_filtered"

    # ====== Dataset and Dataloader ======
    train_dataset = CustomDataset(
        image_dir=os.path.join(base_dir, "train_randaug", "color"),
        mask_dir=os.path.join(base_dir, "train_randaug", "label"),
    )

    def custom_collate_fn(batch):
        images, masks, img_names, mask_names = zip(*batch)

        images = torch.stack(images)
        return images, list(masks), list(img_names), list(mask_names)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)

    val_dataset = CustomDataset(
        image_dir=os.path.join(base_dir, "val_resized", "color"),
        mask_dir=os.path.join(base_dir, "val_resized", "label"),
    )
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)

    # ====== Model & Multi-GPU ======
    model = UNet(in_channels=3, out_channels=4)
    # model = EnhancedUNet(in_channels=3, out_channels=4)

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
    
    # ====== ⭐️  Compute class weights for loss balancing  ⭐️ ======
    # 我们只统计类别 0~3（对应最终的标签：0:Cat, 1:Dog, 2:Background, 3:Boundary），
    # mapping规则与CustomDataset中一致：
    #   - 将原始mask先转为 [0,1] float (背景=0，边界=1)
    #   - 然后设 is_background = (mask==0), is_boundary = (mask==1)
    #   - 对于非背景非边界部分，若文件名首字母为大写，则为 Cat (0)，否则为 Dog (1)
    #   - 最后，将背景赋值为 2，边界赋值为 3。
    print("📊 Computing class weights for loss balancing...")
    mask_dir = os.path.join(base_dir, "train_resized", "label")
    num_classes = 4  # 只统计类别 0-3
    class_counts = torch.zeros(num_classes)

    for filename in tqdm(sorted(os.listdir(mask_dir))):
        if filename.endswith(".png"):
            # ⭐️ 使用 PIL 读取 mask，并将其转换为 numpy 数组
            mask = Image.open(os.path.join(mask_dir, filename)).convert("L")
            mask_np = np.array(mask).astype(np.float32) / 255.0  # 归一化到 [0,1]
            # 根据映射规则生成最终 label:
            # 初始化输出（默认值0）
            mapped = np.zeros_like(mask_np, dtype=np.int64)
            is_background = (mask_np == 0.0)
            is_boundary = (mask_np == 1.0)
            is_catdog = (~is_background) & (~is_boundary)
            # 根据文件名判断动物类别
            if filename[0].isupper():
                mapped[is_catdog] = 0  # Cat
            else:
                mapped[is_catdog] = 1  # Dog
            mapped[is_background] = 2  # Background
            mapped[is_boundary] = 3    # Boundary
            # 累计各类像素数
            for cls in range(num_classes):
                class_counts[cls] += (mapped == cls).sum()
    print("✅ Class pixel counts (for classes 0-3):", class_counts.tolist())
    
    # 使用倒数作为权重
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum()  # normalize
    print("🎯 Class weights:", weights.tolist())
    # 注意：由于模型预测只有 4 类，因此 loss_fn 的 weight 参数应为长度为4的张量

    # ====== Training config ======
    NUM_EPOCHS = 500
    PRINT_INTERVAL = 10
    BEST_MODEL_PATH = "/home/s2103701/Model/best_unet_500_epochs_baseline_aug.pth"
    best_val_loss = float("inf")
    patience = 10
    early_stop_counter = 0

    # loss_fn = nn.CrossEntropyLoss(ignore_index = 4)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # ⭐️ 设置 CrossEntropyLoss, 使用 ignore_index=4 来忽略 unknown（ground truth中的值为4） 
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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

        # ✅ 4. Early Stopping 判断
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0  # 重置 counter
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"💾 Saved new best model at epoch {epoch+1} (val loss: {avg_val_loss:.4f})")
        else:
            early_stop_counter += 1
            print(f"⏳ No improvement. Early stop counter: {early_stop_counter}/{patience}")

            if early_stop_counter >= patience:
                print(f"🛑 Early stopping triggered at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
                break

    print("🎉 Training complete. Best validation loss:", best_val_loss)
