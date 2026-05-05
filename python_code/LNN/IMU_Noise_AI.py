import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# -------------------- 仿真数据生成（单 IMU 带复杂误差） --------------------
def generate_imu_with_errors(num_samples=20000, dt=0.01):
    """
    生成真值运动轨迹，并模拟一个有误差的 IMU。
    返回: true (N,6), imu_raw (N,6)
    每一行六维为 [wx, wy, wz, ax, ay, az]
    """
    t = np.arange(num_samples) * dt

    # 真值角速度（复杂运动）
    wx_true = 0.5 * np.sin(2 * np.pi * 0.1 * t) + 0.2 * np.sin(2 * np.pi * 0.5 * t)
    wy_true = 0.3 * np.cos(2 * np.pi * 0.2 * t) + 0.1 * np.sin(2 * np.pi * 0.3 * t)
    wz_true = 0.1 * np.sin(2 * np.pi * 0.05 * t) + 0.05 * np.cos(2 * np.pi * 0.4 * t)
    # 真值加速度（含重力）
    ax_true = 0.2 * np.sin(2 * np.pi * 0.15 * t) + 0.1 * np.cos(2 * np.pi * 0.25 * t)
    ay_true = 0.2 * np.cos(2 * np.pi * 0.25 * t) + 0.1 * np.sin(2 * np.pi * 0.35 * t)
    az_true = 9.8 + 0.1 * np.sin(2 * np.pi * 0.1 * t) + 0.05 * np.cos(2 * np.pi * 0.2 * t)

    true = np.stack([wx_true, wy_true, wz_true, ax_true, ay_true, az_true], axis=1)

    # ---------- 复杂误差模型 ----------
    # 1. 常值偏置
    bias = np.array([0.02, -0.015, 0.01, 0.03, -0.02, 0.005])

    # 2. 尺度因子误差（偏离1）
    scale = np.array([1.02, 0.98, 1.01, 0.99, 1.015, 0.985])

    # 3. 非正交耦合矩阵（小量交叉影响）
    misalign = np.array([
        [1.0,   0.002, -0.001, 0.0,   0.0,   0.0  ],
        [-0.001, 1.0,   0.001, 0.0,   0.0,   0.0  ],
        [0.001, -0.001, 1.0,   0.0,   0.0,   0.0  ],
        [0.0,   0.0,   0.0,   1.0,   0.003, -0.002],
        [0.0,   0.0,   0.0,  -0.002, 1.0,   0.001],
        [0.0,   0.0,   0.0,   0.001, -0.001, 1.0  ]
    ])

    # 4. 温度/运动相关漂移（低频变化）
    drift = np.zeros((num_samples, 6))
    drift[:, 0] = 0.01 * np.sin(2 * np.pi * 0.01 * t)           # wx 慢漂
    drift[:, 1] = 0.008 * np.cos(2 * np.pi * 0.015 * t)         # wy
    drift[:, 2] = 0.005 * np.sin(2 * np.pi * 0.02 * t)          # wz
    drift[:, 3] = 0.02 * np.sin(2 * np.pi * 0.008 * t)          # ax
    drift[:, 4] = 0.015 * np.cos(2 * np.pi * 0.012 * t)         # ay
    drift[:, 5] = 0.01 * np.sin(2 * np.pi * 0.018 * t)          # az

    # 5. 随机噪声
    noise_std = 0.05
    noise = np.random.randn(num_samples, 6) * noise_std

    # 应用误差：raw = (true * scale) @ misalign + bias + drift + noise
    imu_raw = true * scale                     # 尺度误差
    imu_raw = imu_raw @ misalign.T             # 交叉耦合
    imu_raw += bias + drift + noise            # 偏置、漂移、噪声

    return true, imu_raw

# -------------------- 数据集类 --------------------
class IMUCorrectionDataset(Dataset):
    def __init__(self, true_data, raw_data, window_size=1):
        """
        window_size: 若为1则逐点处理；>1 则取历史窗口作为输入（供时序模型用）
        """
        self.true = torch.tensor(true_data, dtype=torch.float32)
        self.raw = torch.tensor(raw_data, dtype=torch.float32)
        self.window = window_size
        self.valid_len = len(true_data) - window_size + 1

    def __len__(self):
        return self.valid_len

    def __getitem__(self, idx):
        if self.window == 1:
            x = self.raw[idx]
            y = self.true[idx]
        else:
            x = self.raw[idx:idx+self.window].flatten()  # (window*6,)
            y = self.true[idx+self.window-1]             # 窗口最后一个时刻的真值
        return x, y

# -------------------- 误差校正网络（MLP 逐点）--------------------
class IMUCorrectionMLP(nn.Module):
    def __init__(self, input_dim=6, output_dim=6, hidden_dims=[64, 128, 64]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # 直接回归修正后的值
        return self.net(x)

# -------------------- 时序校正网络（1D CNN，适合嵌入式部署）--------------------
class IMUCorrectionCNN(nn.Module):
    def __init__(self, window_size=10, feature_dim=6, output_dim=6):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=feature_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, window*6) -> (batch, window, 6) -> (batch, 6, window)
        batch = x.size(0)
        x = x.view(batch, -1, 6).transpose(1, 2)  # (batch, 6, window)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)              # (batch, 32)
        x = self.fc(x)                            # (batch, 6)
        return x

# -------------------- 训练函数 --------------------
def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cpu'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
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
        avg_train = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
        avg_val = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_model_state = model.state_dict().copy()

        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train:.6f}, Val Loss: {avg_val:.6f}')

    model.load_state_dict(best_model_state)
    return train_losses, val_losses

# -------------------- 主程序 --------------------
if __name__ == '__main__':
    # 生成数据
    true, imu_raw = generate_imu_with_errors(num_samples=20000, dt=0.01)

    # 划分训练/验证/测试
    train_size = int(0.7 * len(true))
    val_size = int(0.15 * len(true))
    test_size = len(true) - train_size - val_size

    train_true, train_raw = true[:train_size], imu_raw[:train_size]
    val_true, val_raw = true[train_size:train_size+val_size], imu_raw[train_size:train_size+val_size]
    test_true, test_raw = true[-test_size:], imu_raw[-test_size:]

    # 使用逐点 MLP（也可以改为 CNN 窗口模型）
    use_cnn = False
    window = 10 if use_cnn else 1

    train_dataset = IMUCorrectionDataset(train_true, train_raw, window_size=window)
    val_dataset   = IMUCorrectionDataset(val_true, val_raw, window_size=window)
    test_dataset  = IMUCorrectionDataset(test_true, test_raw, window_size=window)

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if use_cnn:
        model = IMUCorrectionCNN(window_size=window, feature_dim=6, output_dim=6)
    else:
        model = IMUCorrectionMLP(input_dim=6, output_dim=6)

    print("Training correction model...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=50, device=device)

    # 保存模型
    torch.save(model.state_dict(), 'imu_correction_model.pth')
    print("Model saved as 'imu_correction_model.pth'")

    # 测试集评估
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0
    all_preds = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += criterion(pred, y).item() * x.size(0)
            all_preds.append(pred.cpu().numpy())
    test_mse = test_loss / len(test_dataset)
    all_preds = np.concatenate(all_preds, axis=0)

    # 对比原始误差与修正后误差
    # 注意：如果用了窗口，预测值长度会少 (window-1) 个点，这里只对比有效部分
    if window > 1:
        compare_len = len(test_true) - window + 1
        test_true_align = test_true[window-1:]
        test_raw_align  = test_raw[window-1:]
    else:
        compare_len = len(test_true)
        test_true_align = test_true
        test_raw_align  = test_raw

    raw_mse = np.mean((test_raw_align - test_true_align) ** 2)
    print(f'Test MSE (Raw IMU):   {raw_mse:.6f}')
    print(f'Test MSE (Corrected): {test_mse:.6f}')
    print(f'Improvement: {(raw_mse - test_mse) / raw_mse * 100:.2f}%')

    # 绘制训练曲线
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Training Curves')
    plt.show()

    # 可视化修正效果（角速度 X 轴）
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(test_true_align[:500, 0], label='True wx', linewidth=2)
    plt.plot(test_raw_align[:500, 0], alpha=0.7, label='Raw wx')
    plt.plot(all_preds[:500, 0], label='Corrected wx', linestyle='--')
    plt.legend()
    plt.title('Angular Velocity X Correction')
    plt.subplot(1,2,2)
    plt.plot(test_true_align[:500, 3], label='True ax', linewidth=2)
    plt.plot(test_raw_align[:500, 3], alpha=0.7, label='Raw ax')
    plt.plot(all_preds[:500, 3], label='Corrected ax', linestyle='--')
    plt.legend()
    plt.title('Acceleration X Correction')
    plt.tight_layout()
    plt.show()

    # 可选：导出 ONNX（用于 MCU 部署）
    try:
        import torch.onnx
        dummy = torch.randn(1, 6 if not use_cnn else window*6, device='cpu')
        model.to('cpu')
        model.eval()
        torch.onnx.export(model, dummy, "imu_correction.onnx",
                          input_names=['imu_raw'], output_names=['imu_corrected'],
                          opset_version=11, dynamic_axes={'imu_raw': {0: 'batch'}, 'imu_corrected': {0: 'batch'}})
        print("ONNX model exported as 'imu_correction.onnx'")
    except Exception as e:
        print("ONNX export failed:", e)