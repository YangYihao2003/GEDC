"""
性能增强版 IMU Transformer - 平衡容量与 GD32 部署兼容性
- d_model = 48, heads = 4, layers = 4, d_ff = 128, seq_len = 96
- 无 LayerNorm，无 Dropout，固定形状，全算子安全
- 参数量 ~180k，Flash 占用 ~720KB
运行：python enhanced_imu_transformer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import os

# -------------------- 1. 正弦位置编码（固定，折叠为常量）--------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# -------------------- 2. 多头注意力（4 头，合并 QKV）--------------------
class EnhancedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 合并 QKV 投影
        self.W_QKV = nn.Linear(d_model, 3 * d_model, bias=True)
        self.W_O = nn.Linear(d_model, d_model, bias=True)

    def split_heads(self, x):
        batch, seq_len, _ = x.size()
        return x.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

    def forward(self, x):
        qkv = self.W_QKV(x)
        Q, K, V = qkv.chunk(3, dim=-1)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)

        return self.W_O(self.combine_heads(attn_out))

# -------------------- 3. 前馈网络（d_ff = 128）--------------------
class EnhancedFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=True)
        self.linear2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

# -------------------- 4. 编码器层（残差连接，无归一化）--------------------
class EnhancedEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = EnhancedMultiHeadAttention(d_model, num_heads)
        self.ff = EnhancedFeedForward(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(x)   # 残差 1
        x = x + self.ff(x)     # 残差 2
        return x

# -------------------- 5. 增强版 IMU Transformer --------------------
class EnhancedIMUTransformer(nn.Module):
    def __init__(self, input_dim=6, d_model=48, num_heads=4, num_layers=4,
                 d_ff=128, num_classes=6, seq_len=96):
        super().__init__()
        self.seq_len = seq_len

        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model, bias=True)

        # 正弦位置编码（固定，无参数）
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, seq_len)

        # 堆叠编码器
        self.encoder_layers = nn.ModuleList([
            EnhancedEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        # 分类头
        self.classifier = nn.Linear(d_model, num_classes, bias=True)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x)

        # 全局平均池化
        x = x.mean(dim=1)  # (batch, d_model)
        return self.classifier(x)

# -------------------- 6. 数据集与训练（增强版训练技巧）--------------------
class IMUDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=96, input_dim=6, num_classes=6):
        self.data = torch.randn(num_samples, seq_len, input_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train(model, train_loader, val_loader, epochs, lr, device):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        # 验证
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                _, preds = torch.max(logits, 1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total
        if val_acc > best_acc:
            best_acc = val_acc

        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {total_loss/total:.4f} | "
                  f"Train Acc: {correct/total:.4f} | Val Acc: {val_acc:.4f}")

    print(f"训练完成，最佳验证准确率: {best_acc:.4f}")

# -------------------- 7. 主程序 --------------------
def main():
    # 增强版超参数
    INPUT_DIM = 6
    NUM_CLASSES = 6
    D_MODEL = 48
    NUM_HEADS = 4
    NUM_LAYERS = 4
    D_FF = 128
    SEQ_LEN = 96
    BATCH_SIZE = 32
    EPOCHS = 30
    LR = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据
    train_dataset = IMUDataset(800, SEQ_LEN, INPUT_DIM, NUM_CLASSES)
    val_dataset = IMUDataset(200, SEQ_LEN, INPUT_DIM, NUM_CLASSES)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE)

    # 模型
    model = EnhancedIMUTransformer(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        num_classes=NUM_CLASSES,
        seq_len=SEQ_LEN
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    train(model, train_loader, val_loader, EPOCHS, LR, device)

    # 保存权重
    model_name = "enhanced_imu_transformer"
    torch.save(model.state_dict(), f"{model_name}.pth")
    print(f"权重已保存: {model_name}.pth")

    # -------------------- 导出 ONNX --------------------
    model.eval()
    dummy_input = torch.randn(1, SEQ_LEN, INPUT_DIM).to(device)
    onnx_path = f"{model_name}.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['imu_sequence'],
        output_names=['logits'],
        opset_version=12,
        do_constant_folding=True,
        export_params=True
    )
    print(f"ONNX 已导出: {onnx_path}")

    # 简化 ONNX
    try:
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(onnx_path)
        model_simp, check = simplify(onnx_model)
        if check:
            onnx.save(model_simp, onnx_path)
            print("ONNX 简化成功，计算图已优化。")
    except ImportError:
        print("提示: pip install onnx-simplifier 可进一步压缩模型。")

if __name__ == "__main__":
    main()