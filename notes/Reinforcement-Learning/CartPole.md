CartPole 是 OpenAI Gym 提供的经典强化学习环境之一。它模拟了一个倒立摆（Inverted Pendulum）问题，其中小车（Cart）在轨道上左右移动，以保持一根自由旋转的杆（Pole）平衡。智能体（Agent）的目标是在重力作用下，通过左右移动小车，使杆保持直立尽可能长的时间。

### **环境细节（CartPole-v1）**：

- **状态空间（Observation Space）**：
    - 4 维向量：
        1. 小车位置（Cart Position）
        2. 小车速度（Cart Velocity）
        3. 杆的角度（Pole Angle）
        4. 杆的角速度（Pole Angular Velocity）
- **动作空间（Action Space）**：
    - 2 个离散动作：
        1. `0`：向左施加力
        2. `1`：向右施加力
- **奖励（Reward）**：
    - 每步奖励 +1，目标是让杆尽量长时间保持直立。
- **终止条件（Done）**：
    1. 杆的角度超过 ±12°（约 0.209 弧度）。
    2. 小车的位移超过 ±2.4 个单位（超出轨道）。
    3. 最高 500 步后自动结束（v1 版本的最大步数）。

这个环境是强化学习的入门任务之一，适用于**策略梯度方法（Policy Gradient）**、**Q-learning**、**DQN** 等强化学习算法的研究。