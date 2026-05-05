"""
可调节规模的 IMU Transformer 误差回归模型（窗口级误差估计）
输入：一段长度为 100 的六轴 IMU 序列
输出：该窗口的一个 6 维误差向量（代表整个窗口的误差特性）
通过修改 Config 类中的参数，精确控制 ONNX 大小
运行：python imu_error_scalable_window.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import os


# ==================== 可调节配置（修改这里） ====================
class Config:
    # 输入输出维度
    INPUT_DIM = 6          # 六轴 IMU
    OUTPUT_DIM = 6         # 一个 6 维误差向量（整段窗口的误差估计）
    SEQ_LEN = 100          # 输入窗口的长度（时间步数）
    MAX_SEQ_LEN = 200      # 位置编码最大支持长度（推理时可输入 ≤200 的任意长度）

    # 模型规模参数
    D_MODEL = 36           # 特征维度
    NUM_LAYERS = 2         # 编码器层数
    NUM_HEADS = 4          # 注意力头数（需能被 D_MODEL 整除）
    D_FF = 128             # 前馈网络中间维度

    # 训练参数
    BATCH_SIZE = 32
    EPOCHS = 200
    LR = 1e-3
# =============================================================


# -------------------- 正弦位置编码 --------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                     # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# -------------------- 多头注意力（合并 QKV） --------------------
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


# -------------------- 前馈网络 --------------------
class ScalableFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=True)
        self.linear2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


# -------------------- 编码器层 --------------------
class ScalableEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = ScalableMultiHeadAttention(d_model, num_heads)
        self.ff = ScalableFeedForward(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x


# -------------------- 窗口级误差回归模型 --------------------
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
        # 全局平均池化 → 将序列长度维度压缩，只保留特征
        self.pool = nn.AdaptiveAvgPool1d(1)      # 作用在 seq_len 维度上
        self.regressor = nn.Linear(d_model, output_dim, bias=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)                   # (B, seq_len, d_model)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        # 将 (B, seq_len, d_model) 转为 (B, d_model, seq_len)，在 seq_len 维度做平均池化
        x = x.transpose(1, 2)                    # (B, d_model, seq_len)
        x = self.pool(x).squeeze(-1)             # (B, d_model)
        return self.regressor(x)                 # (B, output_dim=6)


# -------------------- 模拟数据集（窗口级误差） --------------------
class IMUWindowErrorDataset(Dataset):
    def __init__(self, num_samples, seq_len, input_dim, output_dim):
        # 原始 IMU 数据：随机生成
        self.raw_data = torch.randn(num_samples, seq_len, input_dim) * 2.0
        # 模拟窗口平均误差（真实误差可能变化缓慢，这里用随机偏置 + 缓慢漂移近似）
        t = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1)
        drift = torch.sin(2 * math.pi * 0.05 * t) * 0.3   # 缓慢变化
        # 逐点误差由偏置+漂移+非线性+噪声组成
        pointwise_error = drift + 0.1 * self.raw_data ** 2 + torch.randn_like(self.raw_data) * 0.05
        # 对窗口内所有时刻的误差求平均作为标签（也可取末端误差，按需修改）
        self.window_error = pointwise_error.mean(dim=1)   # (num_samples, 6)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        return self.raw_data[idx], self.window_error[idx]


# -------------------- 训练函数 --------------------
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
            pred = model(x)                  # (B, 6)
            loss = criterion(pred, y)       # (B, 6)
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

    # 数据集（窗口级误差）
    train_dataset = IMUWindowErrorDataset(800, cfg.SEQ_LEN, cfg.INPUT_DIM, cfg.OUTPUT_DIM)
    val_dataset = IMUWindowErrorDataset(200, cfg.SEQ_LEN, cfg.INPUT_DIM, cfg.OUTPUT_DIM)
    train_loader = DataLoader(train_dataset, cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, cfg.BATCH_SIZE)

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
    estimated_onnx_kb = total_params * 4 / 1024 * 1.15
    print(f"模型参数量: {total_params:,}")
    print(f"预计 ONNX 大小: {estimated_onnx_kb:.1f} KB")
    print(f"位置编码最大支持序列长度: {cfg.MAX_SEQ_LEN}")

    train_model(model, train_loader, val_loader, cfg.EPOCHS, cfg.LR, device)

    # 保存权重
    model_name = "imu_error_scalable_window"
    torch.save(model.state_dict(), f"{model_name}.pth")
    print(f"权重已保存: {model_name}.pth")

    # 导出 ONNX（动态 batch，无 seq_length 输出）
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
            'imu_sequence': {0: 'batch_size'},   # 只让 batch 可变，seq_length 固定为 100
            'error_window': {0: 'batch_size'}
        }
    )
    print(f"ONNX 已导出: {onnx_path}")
    onnx_size = os.path.getsize(onnx_path)
    print(f"实际 ONNX 文件大小: {onnx_size / 1024:.1f} KB")

    # 可选简化
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


if __name__ == "__main__":
    main()