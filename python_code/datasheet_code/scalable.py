"""
generate_imu_pointwise_error_csv.py
生成单点 IMU 误差回归数据集
每一行：6维IMU测量值 + 6维真实误差（即该时刻的偏置+尺度偏离）
输出文件：train.csv, val.csv, test.csv
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import os

# ==================== 配置参数 ====================
SIMULATION_TIME = 1000.0        # 总仿真时间（秒）
SAMPLE_RATE = 100.0            # 采样频率（Hz）
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 误差模型参数
USE_BIAS = True                # 是否模拟偏置误差
USE_SCALE = True               # 是否模拟尺度误差
USE_NOISE = True               # 是否添加高斯噪声

# 误差变化速度（随机游走的步长标准差）
BIAS_WALK_STD = 0.005          # 偏置变化速度 (m/s² 或 rad/s)
SCALE_WALK_STD = 0.002         # 尺度因子变化速度
NOISE_LEVEL = 0.01             # 高斯噪声标准差

# 运动参数（与之前仿真一致）
def get_true_motion(t):
    """返回载体在世界系下的位置、速度、加速度、欧拉角、欧拉角速度、体轴角速度"""
    x = 5 * np.sin(0.5 * t)
    y = 3 * np.sin(1.0 * t)
    z = 0
    pos = np.array([x, y, z])
    vx = 5 * 0.5 * np.cos(0.5 * t)
    vy = 3 * 1.0 * np.cos(1.0 * t)
    vz = 0
    vel = np.array([vx, vy, vz])
    ax = -5 * (0.5**2) * np.sin(0.5 * t)
    ay = -3 * (1.0**2) * np.sin(1.0 * t)
    az = 0
    acc_world = np.array([ax, ay, az])
    roll = 15 * np.sin(0.8 * t) * (np.pi/180.0)
    pitch = 10 * np.sin(0.6 * t) * (np.pi/180.0)
    yaw = 30 * np.sin(0.4 * t) * (np.pi/180.0)
    euler = np.array([roll, pitch, yaw])
    roll_dot = 15 * 0.8 * np.cos(0.8 * t) * (np.pi/180.0)
    pitch_dot = 10 * 0.6 * np.cos(0.6 * t) * (np.pi/180.0)
    yaw_dot = 30 * 0.4 * np.cos(0.4 * t) * (np.pi/180.0)
    euler_dot = np.array([roll_dot, pitch_dot, yaw_dot])
    cr, cp = np.cos(roll), np.cos(pitch)
    sr, sp = np.sin(roll), np.sin(pitch)
    J = np.array([[1, 0, -sp], [0, cr, sr*cp], [0, -sr, cr*cp]])
    ang_vel_body = J @ euler_dot
    return pos, vel, acc_world, euler, euler_dot, ang_vel_body

def compute_ideal_imu(t):
    """计算 Body 系原点处的理想 IMU 测量值（6维：角速度+比力）"""
    _, _, acc_world, euler, _, ang_vel_body = get_true_motion(t)
    r_b2w = R.from_euler('xyz', euler).as_matrix()
    r_w2b = r_b2w.T
    gravity_world = np.array([0, 0, -9.81])
    gravity_body = r_w2b @ gravity_world
    acc_body_world = r_w2b @ acc_world
    specific_force = acc_body_world - gravity_body
    return np.concatenate([ang_vel_body, specific_force])   # shape (6,)

def generate_error_trajectory(n_steps, walk_std):
    """生成随机游走误差轨迹（偏置或尺度因子），初始为0，返回 (n_steps, 6)"""
    errors = np.zeros((n_steps, 6))
    for i in range(1, n_steps):
        errors[i] = errors[i-1] + np.random.normal(0, walk_std, 6)
    return errors

# ==================== 主生成流程 ====================
def main():
    # 1. 时间轴
    dt = 1.0 / SAMPLE_RATE
    num_samples = int(SIMULATION_TIME * SAMPLE_RATE)
    time = np.arange(num_samples) * dt

    # 2. 理想 IMU 序列
    print("生成理想 IMU 序列...")
    ideal_imu = np.array([compute_ideal_imu(ti) for ti in time])  # (N, 6)

    # 3. 生成时变误差
    n = len(ideal_imu)
    bias = generate_error_trajectory(n, BIAS_WALK_STD) if USE_BIAS else np.zeros((n, 6))
    scale = generate_error_trajectory(n, SCALE_WALK_STD) if USE_SCALE else np.zeros((n, 6))
    scale = 1.0 + scale   # 尺度因子围绕 1 变化

    # 4. 添加误差得到测量值
    noisy_imu = ideal_imu.copy()
    if USE_SCALE:
        noisy_imu = noisy_imu * scale
    if USE_BIAS:
        noisy_imu = noisy_imu + bias
    if USE_NOISE:
        noisy_imu += np.random.normal(0, NOISE_LEVEL, noisy_imu.shape)

    # 5. 真实误差向量（偏置 + 尺度偏离）
    true_error = bias.copy()
    if USE_SCALE:
        true_error += (scale - 1.0)

    # 6. 直接使用每个时间点的数据，不进行窗口化
    # 每行：6个IMU测量值 + 6个误差标签
    data = np.hstack([noisy_imu, true_error])   # (N, 12)

    # 7. 划分数据集（按时间顺序划分，不打乱以保持时间连续性）
    train_end = int(TRAIN_RATIO * n)
    val_end = train_end + int(VAL_RATIO * n)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # 8. 保存为 CSV
    columns = ['gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z',
               'err_gyro_x', 'err_gyro_y', 'err_gyro_z', 'err_acc_x', 'err_acc_y', 'err_acc_z']

    os.makedirs("imu_pointwise_dataset", exist_ok=True)
    pd.DataFrame(train_data, columns=columns).to_csv("imu_pointwise_dataset/train.csv", index=False)
    pd.DataFrame(val_data, columns=columns).to_csv("imu_pointwise_dataset/val.csv", index=False)
    pd.DataFrame(test_data, columns=columns).to_csv("imu_pointwise_dataset/test.csv", index=False)

    print(f"单点数据集已保存至 imu_pointwise_dataset/")
    print(f"总样本数: {n}")
    print(f"训练集: {train_data.shape[0]} 行")
    print(f"验证集: {val_data.shape[0]} 行")
    print(f"测试集: {test_data.shape[0]} 行")
    print("列名:", columns[:6], "|", columns[6:])

    # 9. 可视化示例（第一条数据）
    import matplotlib.pyplot as plt
    sample = train_data[0]
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.bar(range(6), sample[:6])
    plt.title("IMU Measurement (1 sample)")
    plt.xticks(range(6), columns[:6], rotation=45)
    plt.subplot(1,2,2)
    plt.bar(range(6), sample[6:])
    plt.title("True Error")
    plt.xticks(range(6), columns[6:], rotation=45)
    plt.tight_layout()
    plt.savefig("imu_pointwise_dataset/sample.png")
    plt.show()

if __name__ == "__main__":
    main()