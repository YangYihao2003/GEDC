import numpy as np
import pandas as pd
import os

# ==================== 配置参数（与训练时的 Config 一致） ====================
SEQ_LEN = 100           # 时间窗口长度
INPUT_DIM = 6          # 六轴 IMU (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
NUM_SAMPLES = 5        # 生成的样本数量（每条样本是一个完整的输入窗口）

# 输出文件
CSV_FILE = "imu_dataset.csv"
EXCEL_FILE = "imu_dataset.xlsx"

# ==================== 1. 生成模拟 IMU 数据 ====================
def generate_imu_samples(num_samples=NUM_SAMPLES, seq_len=SEQ_LEN):
    """
    生成接近真实运动的 IMU 模拟数据。
    每个样本是一个 (seq_len, 6) 的 numpy 数组。
    加速度部分包含正弦振动和恒定重力，角速度部分包含绕 Z 轴旋转。
    """
    all_samples = []
    t = np.linspace(0, 2 * np.pi, seq_len)  # 一个运动周期

    for i in range(num_samples):
        # 为每个样本加入不同的随机相位和幅度变化
        phase_a = np.random.uniform(0, 2*np.pi)
        phase_g = np.random.uniform(0, 2*np.pi)
        amp_acc = np.random.uniform(0.8, 2.0)
        amp_gyro = np.random.uniform(0.5, 2.5)

        sample = np.zeros((seq_len, INPUT_DIM), dtype=np.float32)

        # 加速度 (单位可理解为 m/s^2，模拟实际传感器量程)
        sample[:, 0] = amp_acc * np.sin(t + phase_a)          # acc_x
        sample[:, 1] = 0.3 * np.cos(2*t + phase_a)            # acc_y 小幅变化
        sample[:, 2] = 9.81 * np.ones(seq_len)                 # acc_z 重力
        # 角速度 (单位 °/s 或 rad/s，这里用 rad/s 示例)
        sample[:, 3] = 0.02 * np.random.randn(seq_len)         # gyro_x 噪声
        sample[:, 4] = 0.02 * np.random.randn(seq_len)         # gyro_y 噪声
        sample[:, 5] = amp_gyro * np.sin(t + phase_g)          # gyro_z 正弦旋转

        all_samples.append(sample)

    return np.array(all_samples)  # shape: (num_samples, seq_len, 6)

# ==================== 2. 转换为表格格式（样本ID + 时间步 + 6轴） ====================
def samples_to_dataframe(samples):
    """
    将三维数组 (num_samples, seq_len, 6) 转换为包含以下列的长格式 DataFrame：
    sample_id, timestep, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    方便在 CSV/Excel 中查看和分析。
    """
    num_samples, seq_len, _ = samples.shape
    records = []
    for s in range(num_samples):
        for t in range(seq_len):
            row = [s, t] + samples[s, t, :].tolist()
            records.append(row)
    columns = ["sample_id", "timestep"] + \
              ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    df = pd.DataFrame(records, columns=columns)
    return df

# ==================== 3. 保存文件 ====================
def save_dataset(samples, csv_path=CSV_FILE, excel_path=EXCEL_FILE):
    df = samples_to_dataframe(samples)
    # 保存 CSV
    df.to_csv(csv_path, index=False)
    print(f"CSV 数据集已保存: {csv_path}（共 {len(df)} 行）")

    # 保存 Excel（每个样本一个 sheet）
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for s in range(samples.shape[0]):
            # 提取一个样本的 DataFrame（无 sample_id、timestep 单独列出）
            sample_df = pd.DataFrame(samples[s],
                                     columns=["acc_x", "acc_y", "acc_z",
                                              "gyro_x", "gyro_y", "gyro_z"])
            sample_df.index.name = 'timestep'
            sheet_name = f"sample_{s}"
            sample_df.to_excel(writer, sheet_name=sheet_name, index=True)
    print(f"Excel 数据集已保存: {excel_path}（每个样本一个 sheet）")

# ==================== 主程序 ====================
if __name__ == "__main__":
    samples = generate_imu_samples()
    print(f"生成样本形状: {samples.shape}（{samples.shape[0]} 个样本，每个 {SEQ_LEN} 时间步，{INPUT_DIM} 轴）")

    flat = samples[0].flatten()
    c_array_str = "{" + ", ".join(f"{v:.6f}f" for v in flat) + "};"
    print(c_array_str)
    save_dataset(samples)
    # 可选：预览第一个样本的前 5 行
    print("\n第一个样本的前 5 行预览：")
    preview = pd.DataFrame(samples[0], columns=["acc_x", "acc_y", "acc_z",
                                                "gyro_x", "gyro_y", "gyro_z"])
    preview.index.name = 'timestep'
    print(preview.head())
