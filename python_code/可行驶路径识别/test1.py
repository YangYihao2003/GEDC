import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import cv2

# -------------------- 配置 (极限压缩) --------------------
class Config:
    NUM_CLASSES = 4
    COLOR_MAP = [
        [128, 128, 128],  # 0 可行驶区域
        [0, 0, 255],      # 1 车辆
        [255, 0, 255],    # 2 其他障碍物（原类别5）
        [0, 0, 0]         # 3 背景/天空
    ]
    IMG_SIZE = 64         # ★ 极限输入尺寸：64x64
    OUT_SIZE = 16         # ★ 网络直接输出 16x16 分割图（应用端放大）
    BATCH_SIZE = 4
    EPOCHS = 10
    LR = 5e-4

cfg = Config()

# -------------------- 超级微型分割模型 (SRAM < 100KB) --------------------
class UltraTinyConvBN(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True)
        )

class UltraTinyBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand=2):
        super().__init__()
        hidden = in_ch * expand
        self.use_skip = (stride == 1) and (in_ch == out_ch)
        layers = [
            UltraTinyConvBN(in_ch, hidden, 1),
            UltraTinyConvBN(hidden, hidden, 3, stride, groups=hidden),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_skip:
            return x + self.block(x)
        return self.block(x)

class UltraTinySegNet(nn.Module):
    """参数量 < 8K，运行时峰值内存 < 80 KB（输入64x64）"""
    def __init__(self, num_classes=4):
        super().__init__()
        # 骨干：64 -> 32 -> 16，通道恒为8
        self.stem = UltraTinyConvBN(3, 8, stride=2)                # 32x32x8
        self.block1 = UltraTinyBlock(8, 8, stride=1, expand=2)     # 32x32x8
        self.block2 = UltraTinyBlock(8, 8, stride=2, expand=2)     # 16x16x8
        # 输出头：1x1卷积直接产生类别logits（无上采样）
        self.head = nn.Conv2d(8, num_classes, 1)                    # 16x16x4

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)      # (B, num_classes, 16, 16)

# -------------------- 数据集：标签自动下采样到 16x16 --------------------
class TinySegDataset(Dataset):
    def __init__(self, num_samples=100, img_size=64, out_size=16, num_classes=4):
        self.num_samples = num_samples
        self.img_size = img_size
        self.out_size = out_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.randn(3, self.img_size, self.img_size)
        # 生成原始分辨率掩膜，再下采样到输出尺寸（最近邻，保持类别值）
        full_mask = torch.randint(0, self.num_classes, (self.img_size, self.img_size), dtype=torch.long)
        small_mask = F.interpolate(
            full_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(self.out_size, self.out_size),
            mode='nearest'
        ).squeeze(0).squeeze(0).long()
        return img, small_mask

# -------------------- 颜色可视化工具（自动放大） --------------------
def decode_color_mask(mask, color_map):
    """mask: (H, W) 低分辨率掩膜 -> 直接转为彩色图并放大至显示尺寸"""
    H, W = mask.shape
    color_img = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, color in enumerate(color_map):
        color_img[mask == cls_id] = color
    return color_img

def blend_images(original_bgr, color_mask_rgb, alpha=0.5):
    color_mask_bgr = cv2.cvtColor(color_mask_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(original_bgr, 1 - alpha, color_mask_bgr, alpha, 0)

# -------------------- 训练与导出 --------------------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = UltraTinySegNet(num_classes=cfg.NUM_CLASSES).to(device)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    # 估算峰值内存（float32下）：最大特征图 (32x32x8) * 4 bytes ≈ 32KB，很安全

    train_dataset = TinySegDataset(200, cfg.IMG_SIZE, cfg.OUT_SIZE, cfg.NUM_CLASSES)
    val_dataset = TinySegDataset(50, cfg.IMG_SIZE, cfg.OUT_SIZE, cfg.NUM_CLASSES)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    best_loss = float('inf')
    for epoch in range(cfg.EPOCHS):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)        # (B, C, 16, 16)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        scheduler.step()
        print(f"Epoch {epoch+1:02d}/{cfg.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'ultratiny_best.pth')
            print("  -> saved best model")

    # -------------------- 导出 ONNX（不含上采样） --------------------
    best_model = UltraTinySegNet(num_classes=cfg.NUM_CLASSES).to(device)
    best_model.load_state_dict(torch.load('ultratiny_best.pth', map_location=device))
    best_model.eval()

    dummy_input = torch.randn(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE).to(device)
    onnx_path = 'ultratiny_seg.onnx'
    torch.onnx.export(
        best_model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes=None
    )
    print(f"ONNX model exported to '{onnx_path}'")

    # -------------------- 演示推理（放大显示） --------------------
    model.eval()
    dummy_img = torch.randn(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE).to(device)
    with torch.no_grad():
        output = model(dummy_img)                              # (1,4,16,16)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # (16,16)

    # 将 16x16 预测放大到 64x64 用于可视化
    pred_small = torch.tensor(pred, dtype=torch.float)
    pred_full = F.interpolate(pred_small.view(1,1,cfg.OUT_SIZE,cfg.OUT_SIZE),
                              size=(cfg.IMG_SIZE, cfg.IMG_SIZE), mode='nearest')
    pred_full = pred_full.squeeze().long().cpu().numpy()       # (64,64)

    color_mask = decode_color_mask(pred_full, cfg.COLOR_MAP)  # 64x64 RGB
    # 构建原图（64x64 RGB）
    original_np = (dummy_img[0].permute(1,2,0).cpu().numpy() * 0.5 + 0.5)
    original_np = np.clip(original_np, 0, 1)
    original_bgr = cv2.cvtColor((original_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    blended = blend_images(original_bgr, color_mask, alpha=0.5)
    cv2.imwrite('demo_ultratiny.png', blended)
    print("Demo blended image saved as 'demo_ultratiny.png'")

if __name__ == '__main__':
    train()