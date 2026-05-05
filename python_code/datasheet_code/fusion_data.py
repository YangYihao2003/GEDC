import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# ================= 1. 配置参数 =================
SIMULATION_TIME = 30.0          # 总仿真时间 (秒)
SAMPLE_RATE = 100.0             # 采样频率 (Hz)
DT = 1.0 / SAMPLE_RATE          # 采样间隔 (秒)
num_samples = int(SIMULATION_TIME * SAMPLE_RATE)
time = np.arange(0, SIMULATION_TIME, DT)

# --- IMU特性与噪声参数 (模拟真实传感器) ---
imu_freq = SAMPLE_RATE
acc_noise_sigma = 0.01          # 加速度计噪声 (m/s^2)
gyr_noise_sigma = 0.005         # 陀螺仪噪声 (rad/s)
acc_bias = np.array([0.1, 0.05, -0.1])   # 加速度计零偏
gyr_bias = np.array([0.01, -0.01, 0.005]) # 陀螺仪零偏

# --- 为双IMU定义不同的安装位置 (在Body Frame下) ---
# 假设Body Frame原点在人体中心，X向前，Y向右，Z向下
imu1_pos_body = np.array([0.2, -0.05, 0.0])  # 左侧IMU (Y轴负方向偏移5cm)
imu2_pos_body = np.array([0.2,  0.05, 0.0])  # 右侧IMU (Y轴正方向偏移5cm)

# ================= 2. 定义真实运动轨迹 (World Frame) =================

def get_true_motion(t):
    """
    计算载体在t时刻的真实位置、速度、加速度、姿态角及其变化率
    returns:
        pos, vel, acc_world, euler, euler_dot, ang_vel_body
    """
    # 位置: 在水平面上做 '8' 字形运动
    x = 5 * np.sin(0.5 * t)
    y = 3 * np.sin(1.0 * t)
    z = 0
    pos = np.array([x, y, z])

    # 速度
    vx = 5 * 0.5 * np.cos(0.5 * t)
    vy = 3 * 1.0 * np.cos(1.0 * t)
    vz = 0
    vel = np.array([vx, vy, vz])

    # 加速度
    ax = -5 * (0.5**2) * np.sin(0.5 * t)
    ay = -3 * (1.0**2) * np.sin(1.0 * t)
    az = 0
    acc_world = np.array([ax, ay, az])

    # 姿态 (欧拉角): 缓慢摇摆
    roll = 15 * np.sin(0.8 * t) * (np.pi/180.0)
    pitch = 10 * np.sin(0.6 * t) * (np.pi/180.0)
    yaw = 30 * np.sin(0.4 * t) * (np.pi/180.0)
    euler = np.array([roll, pitch, yaw])

    # 欧拉角变化率
    roll_dot = 15 * 0.8 * np.cos(0.8 * t) * (np.pi/180.0)
    pitch_dot = 10 * 0.6 * np.cos(0.6 * t) * (np.pi/180.0)
    yaw_dot = 30 * 0.4 * np.cos(0.4 * t) * (np.pi/180.0)
    euler_dot = np.array([roll_dot, pitch_dot, yaw_dot])

    # 将欧拉角速率转换为Body Frame下的角速度
    cr, cp = np.cos(roll), np.cos(pitch)
    sr, sp = np.sin(roll), np.sin(pitch)
    J = np.array([[1, 0, -sp], [0, cr, sr*cp], [0, -sr, cr*cp]])
    ang_vel_body = J @ euler_dot
    return pos, vel, acc_world, euler, euler_dot, ang_vel_body

def compute_center_imu_true(t):
    """
    计算Body坐标系原点处的理想IMU数据（无噪声、无偏置、无离心力）
    返回：[wx, wy, wz, ax, ay, az]
    """
    _, _, acc_world, euler, _, ang_vel_body = get_true_motion(t)
    # 旋转矩阵 (body -> world)
    r_b2w = R.from_euler('xyz', euler).as_matrix()
    r_w2b = r_b2w.T
    # 重力在世界系下 [0,0,-9.81] 转到 body 系
    gravity_world = np.array([0, 0, -9.81])
    gravity_body = r_w2b @ gravity_world
    # 载体加速度转到 body 系
    acc_body_world = r_w2b @ acc_world
    # 中心点的比力 = a_body - g_body （无离心力项）
    specific_force = acc_body_world - gravity_body
    # 输出 [wx, wy, wz, ax, ay, az]
    return np.concatenate([ang_vel_body, specific_force])

def simulate_imu_data(imu_pos_body):
    """
    模拟单个IMU的数据（带噪声、偏置、离心力）
    """
    imu_data = []
    for t_i in time:
        pos_w, vel_w, acc_w, euler, euler_dot, ang_vel_body = get_true_motion(t_i)
        r_b2w = R.from_euler('xyz', euler).as_matrix()
        r_w2b = r_b2w.T

        # --- 加速度计测量值 (比力) ---
        gravity_world = np.array([0, 0, -9.81])
        gravity_body = r_w2b @ gravity_world
        acc_body_world = r_w2b @ acc_w

        # 离心力项
        ang_vel_norm = np.linalg.norm(ang_vel_body)
        if ang_vel_norm > 1e-5:
            centrifugal_acc = np.cross(ang_vel_body, np.cross(ang_vel_body, imu_pos_body))
        else:
            centrifugal_acc = np.zeros(3)

        specific_force = (acc_body_world - centrifugal_acc) - gravity_body
        noise_acc = np.random.normal(0, acc_noise_sigma)
        measured_acc = specific_force + acc_bias + noise_acc

        # --- 陀螺仪测量值 ---
        noise_gyr = np.random.normal(0, gyr_noise_sigma)
        measured_gyr = ang_vel_body + gyr_bias + noise_gyr

        imu_data.append([t_i, *measured_acc, *measured_gyr])

    return np.array(imu_data)

# --- 执行仿真 ---
print("正在生成中心IMU真值...")
center_true_list = [compute_center_imu_true(t_i) for t_i in time]
center_true = np.array(center_true_list)   # shape (N, 6)

print("正在生成IMU-1 (左) 数据...")
imu1_raw = simulate_imu_data(imu1_pos_body)
print("正在生成IMU-2 (右) 数据...")
imu2_raw = simulate_imu_data(imu2_pos_body)

# ================= 后处理：构建数据集 =================
imu1_df = pd.DataFrame(imu1_raw, columns=['time', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
imu2_df = pd.DataFrame(imu2_raw, columns=['time', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
center_df = pd.DataFrame(center_true, columns=['gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z'])
center_df.insert(0, 'time', time)  # 添加时间列

# 合并左右IMU数据供训练输入
combined_df = pd.DataFrame({
    'time': imu1_df['time'],
    'imu1_acc_x': imu1_df['acc_x'], 'imu1_acc_y': imu1_df['acc_y'], 'imu1_acc_z': imu1_df['acc_z'],
    'imu1_gyro_x': imu1_df['gyro_x'], 'imu1_gyro_y': imu1_df['gyro_y'], 'imu1_gyro_z': imu1_df['gyro_z'],
    'imu2_acc_x': imu2_df['acc_x'], 'imu2_acc_y': imu2_df['acc_y'], 'imu2_acc_z': imu2_df['acc_z'],
    'imu2_gyro_x': imu2_df['gyro_x'], 'imu2_gyro_y': imu2_df['gyro_y'], 'imu2_gyro_z': imu2_df['gyro_z']
})

# 数据切分 (训练:验证:测试 = 7:1:2)
train_len = int(0.7 * len(combined_df))
val_len = int(0.1 * len(combined_df))

train_df = combined_df[:train_len]
val_df = combined_df[train_len:train_len+val_len]
test_df = combined_df[train_len+val_len:]

# 对应切分中心真值
center_train_df = center_df[:train_len]
center_val_df = center_df[train_len:train_len+val_len]
center_test_df = center_df[train_len+val_len:]

# 保存数据
combined_df.to_csv('dual_imu_data.csv', index=False)
train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)
test_df.to_csv('test.csv', index=False)

center_df.to_csv('center_imu_true.csv', index=False)
center_train_df.to_csv('center_train_true.csv', index=False)
center_val_df.to_csv('center_val_true.csv', index=False)
center_test_df.to_csv('center_test_true.csv', index=False)

print("✅ 数据集生成完成！")
print("   - 左右IMU数据: dual_imu_data.csv (及 train/val/test.csv)")
print("   - 中心IMU真值: center_imu_true.csv (及 center_train/val/test_true.csv)")