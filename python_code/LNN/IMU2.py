import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


# -------------------- 1. 仿真数据生成 --------------------
def generate_imu_data(num_samples=10000, dt=0.01, noise_std1=0.05, noise_std2=0.1,
                      bias1=0.02, bias2=-0.01, scale_error1=1.02, scale_error2=0.98):
    """
    生成真值轨迹（正弦运动）和两个带误差的 IMU 测量值。
    返回: true_data (N,6), imu1_data (N,6), imu2_data (N,6)
    每一行的六维为 [wx, wy, wz, ax, ay, az]
    """
    t = np.arange(num_samples) * dt
    # 真值角速度（绕三轴的正弦变化）
    wx_true = 0.5 * np.sin(2 * np.pi * 0.1 * t)
    wy_true = 0.3 * np.cos(2 * np.pi * 0.2 * t)
    wz_true = 0.1 * np.sin(2 * np.pi * 0.05 * t)
    # 真值加速度（含重力分量简化处理，此处只做运动加速度）
    ax_true = 0.2 * np.sin(2 * np.pi * 0.15 * t)
    ay_true = 0.2 * np.cos(2 * np.pi * 0.25 * t)
    az_true = 9.8 + 0.1 * np.sin(2 * np.pi * 0.1 * t)  # 重力为主

    true = np.stack([wx_true, wy_true, wz_true, ax_true, ay_true, az_true], axis=1)

    # IMU1 误差模型：噪声 + 偏置 + 尺度误差
    noise1 = np.random.randn(num_samples, 6) * noise_std1
    imu1 = true * scale_error1 + bias1 + noise1

    # IMU2 误差模型
    noise2 = np.random.randn(num_samples, 6) * noise_std2
    imu2 = true * scale_error2 + bias2 + noise2

    return true, imu1, imu2


# -------------------- 2. 数据集类 --------------------
class DualIMUDataset(Dataset):
    def __init__(self, true_data, imu1_data, imu2_data):
        self.true = torch.tensor(true_data, dtype=torch.float32)
        self.imu1 = torch.tensor(imu1_data, dtype=torch.float32)
        self.imu2 = torch.tensor(imu2_data, dtype=torch.float32)

    def __len__(self):
        return len(self.true)

    def __getitem__(self, idx):
        # 输入：两个 IMU 拼接 (12,)
        x = torch.cat([self.imu1[idx], self.imu2[idx]], dim=0)
        y = self.true[idx]  # 真值 (6,)
        return x, y


# -------------------- 3. 融合网络模型 --------------------
class IMUFusionMLP(nn.Module):
    def __init__(self, input_dim=12, output_dim=6, hidden_dims=[64, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# 可选：1D CNN 模型（考虑时序窗口）
class IMUFusionCNN(nn.Module):
    def __init__(self, input_channels=2, seq_len=6, output_dim=6):
        # input_channels: 两个 IMU 视为两个通道，每个通道 6 个特征
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, 12) -> 重塑为 (batch, 2, 6)
        x = x.view(x.size(0), 2, 6)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # (batch, 32)
        x = self.fc(x)
        return x


# -------------------- 4. 训练与评估 --------------------
def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cpu'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []

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
        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

    return train_losses, val_losses


# -------------------- 5. 主程序 --------------------
if __name__ == '__main__':
    # -------------------- 数据生成与划分（同上）--------------------
    true, imu1, imu2 = generate_imu_data(num_samples=20000, dt=0.01)

    train_size = int(0.7 * len(true))
    val_size = int(0.15 * len(true))
    test_size = len(true) - train_size - val_size

    train_true, train_imu1, train_imu2 = true[:train_size], imu1[:train_size], imu2[:train_size]
    val_true, val_imu1, val_imu2 = true[train_size:train_size + val_size], imu1[train_size:train_size + val_size], imu2[
                                                                                                                   train_size:train_size + val_size]
    test_true, test_imu1, test_imu2 = true[-test_size:], imu1[-test_size:], imu2[-test_size:]

    train_dataset = DualIMUDataset(train_true, train_imu1, train_imu2)
    val_dataset = DualIMUDataset(val_true, val_imu1, val_imu2)
    test_dataset = DualIMUDataset(test_true, test_imu1, test_imu2)

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IMUFusionMLP()
    print("Training MLP model...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=50, device=device)

    # -------------------- 保存模型 --------------------
    # 方法1：仅保存模型参数（推荐）
    torch.save(model.state_dict(), 'imu_fusion_mlp.pth')
    print("Model state_dict saved as 'imu_fusion_mlp.pth'")

    # 方法2：保存完整模型（包含结构，但跨环境可能有问题）
    # torch.save(model, 'imu_fusion_mlp_full.pth')

    # -------------------- 测试集评估（同上）--------------------
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += criterion(pred, y).item() * x.size(0)
    test_mse = test_loss / len(test_dataset)
    print(f'Test MSE (fusion): {test_mse:.6f}')

    imu1_mse = np.mean((test_imu1 - test_true) ** 2)
    imu2_mse = np.mean((test_imu2 - test_true) ** 2)
    print(f'IMU1 raw MSE: {imu1_mse:.6f}')
    print(f'IMU2 raw MSE: {imu2_mse:.6f}')
    print(
        f'Improvement over best single IMU: {(min(imu1_mse, imu2_mse) - test_mse) / min(imu1_mse, imu2_mse) * 100:.2f}%')

    # -------------------- 加载模型示例（可选）--------------------
    # 重新创建一个相同结构的模型，然后加载参数
    loaded_model = IMUFusionMLP()
    loaded_model.load_state_dict(torch.load('imu_fusion_mlp.pth', map_location=device))
    loaded_model.to(device)
    loaded_model.eval()
    print("Model loaded successfully for inference.")

    # 验证加载后的模型输出与原始模型一致
    sample_x, sample_y = next(iter(test_loader))
    sample_x = sample_x.to(device)
    with torch.no_grad():
        pred_orig = model(sample_x)
        pred_loaded = loaded_model(sample_x)
    assert torch.allclose(pred_orig, pred_loaded), "Loaded model output mismatch!"
    print("Verified: Loaded model produces identical outputs.")

    # ...（后续绘图代码保持不变）...
    # 绘制损失曲线
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Training Curves')
    plt.show()

    # 可视化融合效果（角速度 X 轴示例）
    test_pred = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            test_pred.append(pred)
    test_pred = np.concatenate(test_pred, axis=0)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(test_true[:500, 0], label='True wx', linewidth=2)
    plt.plot(test_imu1[:500, 0], alpha=0.7, label='IMU1 wx')
    plt.plot(test_imu2[:500, 0], alpha=0.7, label='IMU2 wx')
    plt.plot(test_pred[:500, 0], label='Fused wx', linestyle='--')
    plt.legend()
    plt.title('Angular Velocity X Comparison')
    plt.subplot(1, 2, 2)
    plt.plot(test_true[:500, 3], label='True ax', linewidth=2)
    plt.plot(test_imu1[:500, 3], alpha=0.7, label='IMU1 ax')
    plt.plot(test_imu2[:500, 3], alpha=0.7, label='IMU2 ax')
    plt.plot(test_pred[:500, 3], label='Fused ax', linestyle='--')
    plt.legend()
    plt.title('Acceleration X Comparison')
    plt.tight_layout()
    plt.show()
