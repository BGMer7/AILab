## 概述
REINFORCE 是一种基于策略梯度的强化学习算法，通过直接优化策略函数来最大化累积奖励。它属于蒙特卡洛方法，适用于回合制（episodic）任务。

---

## 算法原理
1. **策略参数化** 
   使用神经网络或其他参数化函数表示策略 $\pi(a|s;\theta)$
2. **轨迹采样** 
   通过当前策略生成完整轨迹（状态、动作、奖励序列）
3. **梯度计算** 
   沿增加高回报轨迹概率的方向更新策略参数

---

## 数学推导
**目标函数（期望回报）**：
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \gamma^t r_t \right]
$$

**策略梯度定理**：
$$
\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_{t=0}^T \nabla_\theta \log \pi(a_t|s_t;\theta) \cdot G_t \right]
$$

**折扣回报定义**：
$$
G_t = \sum_{k=t}^T \gamma^{k-t} r_k
$$

---

## 实现步骤
1. 初始化策略网络 $\theta$
2. 运行策略收集完整轨迹 $\{s_0,a_0,r_0,...,s_T,a_T,r_T\}$
3. 计算每个时刻的折扣回报 $G_t$
4. 定义损失函数：
   $$
   \mathcal{L} = -\sum_{t=0}^T \log \pi(a_t|s_t;\theta) \cdot G_t
   $$
5. 执行梯度下降更新：
   $$
   \theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}
   $$

---

## 优缺点分析
### 优点
- 直接优化策略，避免值函数估计偏差
- 适用于连续动作空间
- 理论保证收敛到局部最优

### 缺点
- 高方差（需大量样本）
- 学习效率较低
- 需要完整轨迹后才能更新

---

## 应用场景
- 回合制环境（如围棋、回合制游戏）
- 连续控制任务（如机器人控制）
- 策略结构复杂的场景

---

## 代码示例（PyTorch）
```python
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)  # 修正括号闭合
        )
    
    def forward(self, x):
        return self.fc(x)

# 初始化
policy = PolicyNet(state_dim=4, action_dim=2)
optimizer = optim.Adam(policy.parameters(), lr=0.01)
gamma = 0.99

# 训练循环
for episode in range(1000):
    states, actions, rewards = [], [], []
    state = env.reset()
    
    # 轨迹采样
    while True:
        probs = policy(torch.FloatTensor(state))
        action = torch.multinomial(probs, 1).item()
        next_state, reward, done, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state
        if done: break
    
    # 计算折扣回报
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    # 策略更新
    optimizer.zero_grad()
    loss = 0
    for s, a, G in zip(states, actions, returns):
        prob = policy(torch.FloatTensor(s))
        loss += -torch.log(prob[a]) * G
    loss.backward()
    optimizer.step()
```

## 为什么策略梯度要用对数概率(log prob)？

在策略梯度算法中使用对数概率（log probability）是数学推导的自然结果，主要源于以下原因：

---

### 1. 数学推导的必然性
策略梯度定理的核心公式为：
$$
\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_{t=0}^T \nabla_\theta \log \pi(a_t|s_t;\theta) \cdot G_t \right]
$$

这一结果的推导过程如下：
- **目标函数**：最大化轨迹的期望回报：
  $$
  J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \gamma^t r_t \right]
  $$
- **策略梯度展开**：
  $$
  \nabla_\theta J(\theta) = \mathbb{E} \left[ \left( \sum_{t=0}^T \gamma^t r_t \right) \cdot \nabla_\theta \log p(\tau;\theta) \right]
  $$
  其中 $p(\tau;\theta)$ 是轨迹 $\tau$ 的概率，可分解为：
  $$
  p(\tau;\theta) = \prod_{t=0}^T \pi(a_t|s_t;\theta) \cdot p(s_{t+1}|s_t,a_t)
  $$
- **对数转换**：
  对 $p(\tau;\theta)$ 取对数后，乘法转换为加法：
  $$
  \log p(\tau;\theta) = \sum_{t=0}^T \log \pi(a_t|s_t;\theta) + \log p(s_{t+1}|s_t,a_t)
  $$
  由于环境转移概率 $p(s_{t+1}|s_t,a_t)$ 与策略参数 $\theta$ 无关，其梯度为 $0$，最终得到：
  $$
  \nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_{t=0}^T \nabla_\theta \log \pi(a_t|s_t;\theta) \cdot G_t \right]
  $$

**核心意义**：
对数的引入是数学推导的直接结果，它将概率连乘转换为求和，从而简化了梯度计算。

---

### 数值稳定性
- **概率相乘的数值问题**：
  直接计算策略概率的乘积 $\prod_{t=0}^T \pi(a_t|s_t;\theta)$ 会导致数值下溢（尤其当 $T$ 较大时）。
- **对数转换的优势**：
  对数将概率相乘转换为求和，避免数值下溢问题，同时保持单调性（$\log$ 是单调递增函数）。

---

### 梯度计算的简化
- **链式法则的适配性**：
  深度学习框架（如 PyTorch、TensorFlow）通过自动微分计算梯度。若损失函数设计为：
  $$
  \mathcal{L} = -\sum \log \pi(a_t|s_t;\theta) \cdot G_t
  $$
  框架会直接计算 $\nabla_\theta \log \pi$，这与策略梯度定理的公式完全一致。
- **避免概率倒数**：
  若直接对 $\pi(a_t|s_t;\theta)$ 求梯度，其表达式会包含 $\frac{1}{\pi(a_t|s_t;\theta)}$，当 $\pi$ 接近 $0$ 时会导致数值不稳定。

---

### 与最大似然估计的联系
策略梯度可看作一种**加权最大似然估计**：
- 普通最大似然估计的目标是最大化 $\sum \log \pi(a_t|s_t;\theta)$；
- 策略梯度在此基础上增加了权重 $G_t$，使得高回报的动作获得更大的概率提升。

---

### 总结
| 关键点         | 说明                             |
| -------------- | -------------------------------- |
| **数学必然性** | 对数是推导策略梯度定理的必然结果 |
| **数值稳定性** | 避免概率连乘的下溢问题           |
| **计算兼容性** | 适配深度学习框架的自动微分机制   |
| **优化一致性** | 通过加权最大似然实现策略改进     |

代码示例中的负号（`-torch.log(prob[a]) * G`）是因为深度学习框架默认执行梯度下降（最小化损失），而策略梯度需要最大化回报，因此通过最小化负的加权对数概率来实现梯度上升。