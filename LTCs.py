import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

class LTCCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 可学习参数（论文公式(1)参数）
        self.tau = nn.Parameter(torch.randn(hidden_size))  # 原始时间常数
        self.A = nn.Parameter(torch.randn(hidden_size))    # 新增参数A
        
        # 输入和状态的权重矩阵（对应f的网络）
        self.Wx = nn.Linear(input_size, hidden_size, bias=False)
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b = nn.Parameter(torch.zeros(hidden_size))
        
        # 激活函数使用Sigmoid（论文中f的典型选择）
        self.activation = nn.Sigmoid()

    def forward(self, x, state):
        # 计算门控信号（论文中的f(x,I)项）
        gate = self.activation(
            self.Wx(x) + self.Wh(state) + self.b
        )
        
        # 计算系统时间常数（确保正数）
        tau_sys = 1.0 / (1.0/torch.exp(self.tau) + gate)  # 论文中的τ_sys
        
        # 按公式(1)计算导数
        dxdt = - (1/torch.exp(self.tau) + gate) * state + gate * self.A
        
        return dxdt, tau_sys  # 返回导数和系统时间常数（可选）

class LTCNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dt=0.1):
        super().__init__()
        self.cell = LTCCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dt = dt  # 欧拉法步长

    def forward(self, x, initial_state=None):
        batch_size, seq_len, _ = x.size()
        
        # 初始化隐藏状态
        state = torch.zeros(batch_size, self.cell.hidden_size) if initial_state is None else initial_state
        state = state.to(x.device)
        
        # 存储所有时间步的状态（可选）
        states = []
        
        # 按时间步迭代更新
        for t in range(seq_len):
            xt = x[:, t, :]
            dxdt, _ = self.cell(xt, state)
            state = state + self.dt * dxdt  # 显式欧拉法更新
            states.append(state.unsqueeze(1))
        
        # 仅返回最后状态用于预测
        return self.fc(state)

# 数据生成函数（正弦波+噪声）
def generate_sine_data(seq_length, n_samples, noise=0.01):
    t = np.linspace(0, 8*np.pi, n_samples + seq_length)
    data = np.sin(t) + np.random.normal(0, noise, t.shape)
    
    X = np.array([data[i:i+seq_length] for i in range(n_samples)])
    y = np.array([data[i+seq_length] for i in range(n_samples)])
    
    return (torch.FloatTensor(X).unsqueeze(-1),
            torch.FloatTensor(y).unsqueeze(-1))

# 训练流程（与之前类似，但参数需调整）
if __name__ == "__main__":
    # 数据准备
    X, y = generate_sine_data(seq_length=10, n_samples=200)
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 模型参数
    model = LTCNetwork(input_size=1, hidden_size=32, output_size=1, dt=0.1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # 训练循环
    num_epochs = 100
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        train_losses.append(total_loss/len(train_loader))
        print(f"Epoch {epoch+1} | Loss: {train_losses[-1]:.4f}")



    plt.plot(train_losses, label='Train Loss')
    # plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.title("Training Progress")
    plt.show()
