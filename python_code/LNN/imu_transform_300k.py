"""
imu_window_train_from_csv.py
使用 CSV 格式的单点 IMU 数据训练窗口级 Transformer 误差回归模型
数据集要求：CSV 包含 12 列，顺序为：
    gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z,
    err_gyro_x, err_gyro_y, err_gyro_z, err_acc_x, err_acc_y, err_acc_z
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
import os

# ==================== 配置（与模型保持一致） ====================
class Config:
    INPUT_DIM = 6
    OUTPUT_DIM = 6
    SEQ_LEN = 100               # 窗口长度
    MAX_SEQ_LEN = 200
    D_MODEL = 36
    NUM_LAYERS = 2
    NUM_HEADS = 4
    D_FF = 128
    BATCH_SIZE = 32
    EPOCHS = 1000
    LR = 1e-3
    STRIDE = 50                 # 滑动窗口步长（可调整，避免样本重叠过多）
# =============================================================

# -------------------- 模型定义（与原代码完全相同）--------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class ScalableMultiHeadAttention(nn.Module):
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

class ScalableFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=True)
        self.linear2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class ScalableEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = ScalableMultiHeadAttention(d_model, num_heads)
        self.ff = ScalableFeedForward(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x

class IMUErrorRegressorWindow(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, d_ff, output_dim, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.input_proj = nn.Linear(input_dim, d_model, bias=True)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.encoder_layers = nn.ModuleList([
            ScalableEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Linear(d_model, output_dim, bias=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.regressor(x)

# -------------------- 从 CSV 构建滑动窗口数据集 --------------------
class WindowErrorDatasetFromCSV(Dataset):
    def __init__(self, csv_path, seq_len, stride, input_cols, label_cols):
        """
        csv_path: 单点数据CSV路径
        seq_len: 窗口长度
        stride: 滑动步长
        input_cols: 输入特征列名列表 (6个)
        label_cols: 误差标签列名列表 (6个)
        """
        df = pd.read_csv(csv_path)
        # 提取输入序列 (N, 6) 和误差序列 (N, 6)
        self.inputs = df[input_cols].values.astype(np.float32)
        self.labels = df[label_cols].values.astype(np.float32)
        assert len(self.inputs) == len(self.labels), "输入和标签长度不一致"

        # 生成滑动窗口索引
        self.indices = []
        for start in range(0, len(self.inputs) - seq_len + 1, stride):
            self.indices.append(start)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + Config.SEQ_LEN
        window_input = self.inputs[start:end]          # (SEQ_LEN, 6)
        window_label = self.labels[start:end]          # (SEQ_LEN, 6)
        # 取窗口内误差的平均值作为该窗口的标签
        label_avg = window_label.mean(axis=0)          # (6,)
        return torch.tensor(window_input, dtype=torch.float32), \
            torch.tensor(label_avg, dtype=torch.float32)

# -------------------- 训练函数（与原代码相同）--------------------
def train_model(model, train_loader, val_loader, epochs, lr, device):
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

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    print(f"训练完成，最佳验证损失: {best_val_loss:.6f}")

# -------------------- 主程序 --------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    cfg = Config()

    # CSV 列名（根据你生成的数据）
    input_cols = ['gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z']
    label_cols = ['err_gyro_x', 'err_gyro_y', 'err_gyro_z', 'err_acc_x', 'err_acc_y', 'err_acc_z']

    # 加载训练/验证/测试 CSV（请根据实际路径修改）
    train_dataset = WindowErrorDatasetFromCSV('D:/YYH/哈工程/2026_GEDC/python_code/datasheet_code/imu_pointwise_dataset/train.csv',
                                              cfg.SEQ_LEN, cfg.STRIDE,
                                              input_cols, label_cols)
    val_dataset   = WindowErrorDatasetFromCSV('D:/YYH/哈工程/2026_GEDC/python_code/datasheet_code/imu_pointwise_dataset/val.csv',
                                              cfg.SEQ_LEN, cfg.STRIDE,
                                              input_cols, label_cols)
    test_dataset  = WindowErrorDatasetFromCSV('D:/YYH/哈工程/2026_GEDC/python_code/datasheet_code/imu_pointwise_dataset/test.csv',
                                              cfg.SEQ_LEN, cfg.STRIDE,
                                              input_cols, label_cols)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE)
    test_loader  = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE)

    print(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}, 测试样本数: {len(test_dataset)}")

    # 模型
    model = IMUErrorRegressorWindow(
        input_dim=cfg.INPUT_DIM,
        d_model=cfg.D_MODEL,
        num_heads=cfg.NUM_HEADS,
        num_layers=cfg.NUM_LAYERS,
        d_ff=cfg.D_FF,
        output_dim=cfg.OUTPUT_DIM,
        max_seq_len=cfg.MAX_SEQ_LEN
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 训练
    train_model(model, train_loader, val_loader, cfg.EPOCHS, cfg.LR, device)

    # 保存权重和 ONNX
    model_name = "imu_error_scalable_window"
    torch.save(model.state_dict(), f"{model_name}.pth")
    print(f"权重已保存: {model_name}.pth")

    model.eval()
    dummy_input = torch.randn(1, cfg.SEQ_LEN, cfg.INPUT_DIM).to(device)
    onnx_path = f"{model_name}.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['imu_sequence'],
        output_names=['error_window'],
        opset_version=12,
        do_constant_folding=True,
        export_params=True,
        dynamic_axes={
            'imu_sequence': {0: 'batch_size'},   # batch 可变，seq_len 固定
            'error_window': {0: 'batch_size'}
        }
    )
    print(f"ONNX 已导出: {onnx_path}")
    onnx_size = os.path.getsize(onnx_path)
    print(f"实际 ONNX 文件大小: {onnx_size / 1024:.1f} KB")

    # 可选 ONNX 简化
    try:
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(onnx_path)
        model_simp, check = simplify(onnx_model)
        if check:
            onnx.save(model_simp, onnx_path)
            simp_size = os.path.getsize(onnx_path)
            print(f"简化后 ONNX 大小: {simp_size / 1024:.1f} KB")
    except ImportError:
        pass

    # 测试评估
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += criterion(pred, y).item() * x.size(0)
    test_loss /= len(test_dataset)
    print(f"测试集 MSE: {test_loss:.6f}")

if __name__ == "__main__":
    main()