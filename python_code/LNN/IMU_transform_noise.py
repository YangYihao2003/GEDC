"""
增强版 IMU Transformer 误差回归模型 - 适用于 IMU 校准、漂移估计
输入: (batch, seq_len=96, 6)  原始六轴数据 (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
输出: (batch, seq_len=96, 6)  对应时刻的六轴误差估计值
运行：python imu_error_regression.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np

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
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x

# -------------------- 5. 增强版 IMU Transformer 回归模型 --------------------
class EnhancedIMUTransformerRegressor(nn.Module):
    def __init__(self, input_dim=6, d_model=48, num_heads=4, num_layers=4,
                 d_ff=128, output_dim=6, seq_len=96):
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

        # 回归头：输出每个时间步的误差（与输入维度相同）
        self.regressor = nn.Linear(d_model, output_dim, bias=True)

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

        # 直接输出每个时间步的误差预测
        error = self.regressor(x)   # (batch, seq_len, output_dim)
        return error

# -------------------- 6. 模拟 IMU 误差数据集（回归任务）--------------------
class IMUErrorDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=96, input_dim=6, output_dim=6):
        """
        模拟数据：
        - 原始 IMU 数据：随机生成，模拟人体运动产生的加速度/角速度变化。
        - 真实误差：由原始数据经过非线性函数 + 噪声生成，模拟传感器偏差和漂移。
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 生成随机原始数据（模拟运动信号）
        self.raw_data = torch.randn(num_samples, seq_len, input_dim) * 2.0

        # 生成对应的真实误差（非线性函数 + 噪声）
        # 假设误差与原始信号的平方、交叉项有关，并加入低频漂移
        t = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        drift = torch.sin(2 * math.pi * 0.1 * t) * 0.5                 # 低频漂移
        nonlinear = 0.1 * self.raw_data ** 2 + 0.05 * torch.roll(self.raw_data, shifts=1, dims=1)
        noise = torch.randn(num_samples, seq_len, output_dim) * 0.05
        self.error = nonlinear + drift + noise

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.raw_data[idx], self.error[idx]

# -------------------- 7. 训练与验证函数（回归）--------------------
def train_regression(model, train_loader, val_loader, epochs, lr, device):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item() * x.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    print(f"训练完成，最佳验证损失: {best_val_loss:.6f}")

# -------------------- 8. 主程序 --------------------
def main():
    # 超参数
    INPUT_DIM = 6
    OUTPUT_DIM = 6       # 输出六轴误差
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

    # 数据集
    train_dataset = IMUErrorDataset(num_samples=800, seq_len=SEQ_LEN,
                                    input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
    val_dataset = IMUErrorDataset(num_samples=200, seq_len=SEQ_LEN,
                                  input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE)

    # 模型
    model = EnhancedIMUTransformerRegressor(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        output_dim=OUTPUT_DIM,
        seq_len=SEQ_LEN
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    train_regression(model, train_loader, val_loader, EPOCHS, LR, device)

    # 保存权重
    model_name = "imu_error_regressor"
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
        output_names=['error'],
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