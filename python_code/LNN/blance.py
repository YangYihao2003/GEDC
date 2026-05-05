import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# ---------- 1. LTCCell ----------
class LTCCell(nn.Module):
    def __init__(self, input_size, hidden_size, dt=0.05):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt

        self.w_in = nn.Parameter(torch.randn(input_size, hidden_size) * 0.05)
        self.w_rec = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.05)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.tau = nn.Parameter(torch.ones(hidden_size) * 2.0)
        self.activation = nn.Tanh()

    def forward(self, x, h):
        combined = torch.matmul(x, self.w_in) + torch.matmul(h, self.w_rec) + self.bias
        dh_dt = (-h + self.activation(combined)) / torch.abs(self.tau)
        h_new = h + self.dt * dh_dt
        return h_new


# ---------- 2. LTCLayer ----------
class LTCLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dt=0.05):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = LTCCell(input_size, hidden_size, dt)

    def forward(self, x_seq, h0=None):
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(1)

        batch_size, seq_len, feat_size = x_seq.shape
        assert feat_size == self.input_size, f"特征数错误！期望 {self.input_size}，实际 {feat_size}"

        if h0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x_seq.device)
        else:
            h = h0

        outputs = []
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            h = self.cell(x_t, h)
            outputs.append(h)
        return torch.stack(outputs, dim=1), h


# ---------- 3. 控制器 ----------
class LiquidBalanceController(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=1, dt=0.05):
        super().__init__()
        self.ltc_layer = LTCLayer(input_size, hidden_size, dt)
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, x_seq):
        ltc_out, _ = self.ltc_layer(x_seq)
        final_state = ltc_out[:, -1, :]
        output = self.readout(final_state)
        return output, ltc_out


# ---------- 4. 模拟器 ----------
class BalanceCartSimulator:
    def __init__(self, dt=0.05, mass=0.5, length=0.3, damping=0.1):
        self.dt = dt
        self.g = 9.81
        self.mass = mass
        self.length = length
        self.damping = damping
        self.reset()

    def reset(self, init_theta=0.1, init_x=0.0):
        self.theta = init_theta
        self.theta_dot = 0.0
        self.x = init_x
        self.x_dot = 0.0

    def step(self, torque):
        theta_ddot = (self.g / self.length) * np.sin(self.theta) + \
                     (torque / (self.mass * self.length ** 2)) - \
                     self.damping * self.theta_dot
        self.theta_dot += theta_ddot * self.dt
        self.theta += self.theta_dot * self.dt
        self.x_dot += torque * self.dt / self.mass
        self.x += self.x_dot * self.dt
        return self.get_state()

    def get_state(self):
        return np.array([self.theta, self.theta_dot, self.x, self.x_dot], dtype=np.float32)


# ---------- 5. 数据生成 ----------
def generate_training_data(simulator, num_episodes=1000, seq_length=150):
    X_data, y_data = [], []
    for _ in range(num_episodes):
        init_theta = np.random.uniform(-0.2, 0.2)
        init_x = np.random.uniform(-0.5, 0.5)
        simulator.reset(init_theta, init_x)

        states, actions = [], []
        for _ in range(seq_length):
            state = simulator.get_state()
            torque = -8.0 * state[0] - 1.5 * state[1] - 1.0 * state[2] - 0.3 * state[3]
            torque += np.random.normal(0, 0.05)
            torque = np.clip(torque, -1.0, 1.0)
            states.append(state)
            actions.append([torque])
            simulator.step(torque)
        X_data.append(states)
        y_data.append(actions)
    return np.array(X_data, dtype=np.float32), np.array(y_data, dtype=np.float32)


# ---------- 6. 主程序 ----------
if __name__ == "__main__":
    print("生成训练数据中...")
    simulator = BalanceCartSimulator()
    X_data, y_data = generate_training_data(simulator, num_episodes=500, seq_length=100)  # 可调小快速测试

    # 标准化
    X_mean = X_data.mean(axis=(0, 1)).astype(np.float32)
    X_std = X_data.std(axis=(0, 1)).astype(np.float32) + 1e-8
    y_mean = y_data.mean(axis=(0, 1)).astype(np.float32)
    y_std = y_data.std(axis=(0, 1)).astype(np.float32) + 1e-8

    X_norm = (X_data - X_mean) / X_std
    y_norm = (y_data - y_mean) / y_std

    X_tensor = torch.FloatTensor(X_norm)
    y_tensor = torch.FloatTensor(y_norm)

    model = LiquidBalanceController(input_size=4, hidden_size=64, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    batch_size = 64
    epochs = 200

    print("开始训练...")
    for epoch in range(epochs):
        permutation = torch.randperm(X_tensor.size(0))
        epoch_loss = 0.0

        for i in range(0, X_tensor.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = X_tensor[indices]
            batch_y = y_tensor[indices]

            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y[:, -1, :])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.6e}')

    print("\n训练完成，开始测试...\n")
    torch.save(model.state_dict(), "liquid_balance_controller.pth")
    print("模型参数已保存为 liquid_balance_controller.pth")
    # 🔧 强制复制一份全新的连续数组，彻底杜绝视图/内存问题
    X_mean = np.array(X_mean.flatten()[:4], dtype=np.float32, copy=True)
    X_std = np.array(X_std.flatten()[:4], dtype=np.float32, copy=True)
    y_mean = float(y_mean)
    y_std = float(y_std)

    model.eval()
    test_simulator = BalanceCartSimulator()
    test_simulator.reset(init_theta=0.15, init_x=0.0)
    test_states, test_torques = [], []

    with torch.no_grad():
        for step in range(300):
            state = test_simulator.get_state().astype(np.float32)
            state_norm = (state - X_mean) / (X_std + 1e-8)

            # 直接使用 torch.tensor 从 Python 列表创建，确保元素数量为 4
            state_list = state_norm.tolist()
            state_tensor = torch.tensor(state_list, dtype=torch.float32).view(1, 1, 4)

            output, _ = model(state_tensor)
            torque_norm = output.squeeze().item()
            torque = torque_norm * y_std + y_mean
            torque = np.clip(torque, -1.0, 1.0)

            test_simulator.step(torque)
            test_states.append(state)
            test_torques.append(torque)

    test_states = np.array(test_states)
    test_torques = np.array(test_torques)

    # 绘图
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(test_states[:, 0], 'b-', label='Angle (rad)')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.ylabel('Angle')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(test_states[:, 1], 'r-', label='Angular Velocity (rad/s)')
    plt.ylabel('Angular Velocity')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(test_torques, 'g-', label='Control Torque')
    plt.ylabel('Torque')
    plt.xlabel('Time Step')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# # 在训练循环结束后，测试之前插入
# torch.save(model.state_dict(), "liquid_balance_controller.pth")
# print("模型参数已保存为 liquid_balance_controller.pth")
#
# # 1. 重新创建模型结构（必须与保存时一致）
# model = LiquidBalanceController(input_size=4, hidden_size=64, output_size=1)
#
# # 2. 加载保存的参数
# model.load_state_dict(torch.load("liquid_balance_controller.pth"))
# model.eval()  # 切换到评估模式
#
# # 3. 直接用于推理（假设已有 X_mean, X_std, y_mean, y_std）
# state = test_simulator.get_state()
# state_norm = (state - X_mean) / (X_std + 1e-8)
# state_tensor = torch.tensor(state_norm, dtype=torch.float32).view(1, 1, 4)
#
# with torch.no_grad():
#     output, _ = model(state_tensor)
#     torque = output.squeeze().item() * y_std + y_mean
