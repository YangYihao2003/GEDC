import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- 1. 从 CSV 加载双 IMU 数据和中心真值 --------------------
def load_data(csv_dual_path, csv_center_path):
    """
    读取 dual_imu_data.csv 和 center_imu_true.csv
    返回：true_data (N,6), imu1_data (N,6), imu2_data (N,6)
    其中 imu1/imu2 的顺序为 [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]
    true_data 的顺序为 [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]
    """
    # 读取左右IMU数据
    df_dual = pd.read_csv(csv_dual_path)
    imu1_cols = ['imu1_gyro_x', 'imu1_gyro_y', 'imu1_gyro_z',
                 'imu1_acc_x', 'imu1_acc_y', 'imu1_acc_z']
    imu2_cols = ['imu2_gyro_x', 'imu2_gyro_y', 'imu2_gyro_z',
                 'imu2_acc_x', 'imu2_acc_y', 'imu2_acc_z']
    imu1 = df_dual[imu1_cols].values.astype(np.float32)
    imu2 = df_dual[imu2_cols].values.astype(np.float32)

    # 读取中心真值（包含 time, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z）
    df_center = pd.read_csv(csv_center_path)
    center_cols = ['gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z']
    true = df_center[center_cols].values.astype(np.float32)

    # 可选：检查长度是否一致
    assert len(imu1) == len(true) and len(imu2) == len(true), \
        f"Length mismatch: imu1={len(imu1)}, true={len(true)}"

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

# -------------------- 3. MLP 融合网络 --------------------
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

# -------------------- 4. 训练函数 --------------------
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
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

    return train_losses, val_losses

# -------------------- 5. 导出 ONNX（固定 batch 大小 = 1）--------------------
def export_to_onnx_fixed_batch(model, onnx_path="imu_fusion_model.onnx", device='cpu'):
    model.eval()
    dummy_input = torch.randn(1, 12, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        # 不设置 dynamic_axes，所有维度固定（包括 batch 维度）
    )
    print(f"ONNX model saved to {onnx_path} (fixed batch size = 1)")

# -------------------- 6. 主程序 --------------------
if __name__ == '__main__':
    # ========== 1. 加载数据 ==========
    dual_csv_path = r'D:\YYH\哈工程\2026_GEDC\python_code\datasheet_code\dual_imu_data.csv'
    center_csv_path = r'D:\YYH\哈工程\2026_GEDC\python_code\datasheet_code\center_imu_true.csv'
    true, imu1, imu2 = load_data(dual_csv_path, center_csv_path)

    # ========== 2. 划分数据集 ==========
    train_size = int(0.7 * len(true))
    val_size   = int(0.15 * len(true))
    test_size  = len(true) - train_size - val_size

    train_true, train_imu1, train_imu2 = true[:train_size], imu1[:train_size], imu2[:train_size]
    val_true,   val_imu1,   val_imu2   = true[train_size:train_size+val_size], imu1[train_size:train_size+val_size], imu2[train_size:train_size+val_size]
    test_true,  test_imu1,  test_imu2  = true[-test_size:], imu1[-test_size:], imu2[-test_size:]

    # ========== 3. DataLoader ==========
    batch_size = 256
    train_dataset = DualIMUDataset(train_true, train_imu1, train_imu2)
    val_dataset   = DualIMUDataset(val_true, val_imu1, val_imu2)
    test_dataset  = DualIMUDataset(test_true, test_imu1, test_imu2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    # ========== 4. 训练 ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IMUFusionMLP()
    print("Training MLP model...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=5000, device=device)

    # ========== 5. 保存 PyTorch 权重 ==========
    torch.save(model.state_dict(), 'imu_fusion_mlp.pth')
    print("Model state_dict saved as 'imu_fusion_mlp.pth'")

    # ========== 6. 导出 ONNX（固定 batch） ==========
    export_to_onnx_fixed_batch(model, onnx_path="imu_fusion_model.onnx", device=device)

    # ========== 7. 测试集评估 ==========
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

    # 计算原始 IMU 与中心真值的 MSE（用于对比）
    imu1_mse = np.mean((test_imu1 - test_true) ** 2)
    imu2_mse = np.mean((test_imu2 - test_true) ** 2)
    print(f'IMU1 raw MSE: {imu1_mse:.6f}')
    print(f'IMU2 raw MSE: {imu2_mse:.6f}')
    print(f'Improvement over best single IMU: {(min(imu1_mse, imu2_mse)-test_mse)/min(imu1_mse, imu2_mse)*100:.2f}%')

    # ========== 8. 绘图 ==========
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Training Curves')
    plt.show()

    # 可视化前500点
    test_pred = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            test_pred.append(pred)
    test_pred = np.concatenate(test_pred, axis=0)

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(test_true[:500,0], label='True wx', linewidth=2)
    plt.plot(test_imu1[:500,0], alpha=0.7, label='IMU1 wx')
    plt.plot(test_imu2[:500,0], alpha=0.7, label='IMU2 wx')
    plt.plot(test_pred[:500,0], label='Fused wx', linestyle='--')
    plt.legend()
    plt.title('Angular Velocity X')
    plt.subplot(1,2,2)
    plt.plot(test_true[:500,3], label='True ax', linewidth=2)
    plt.plot(test_imu1[:500,3], alpha=0.7, label='IMU1 ax')
    plt.plot(test_imu2[:500,3], alpha=0.7, label='IMU2 ax')
    plt.plot(test_pred[:500,3], label='Fused ax', linestyle='--')
    plt.legend()
    plt.title('Acceleration X')
    plt.tight_layout()
    plt.show()