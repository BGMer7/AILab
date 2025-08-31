# 背景

## DQN的问题

在原始的 DQN 中，使用同一个网络来选择动作和评估动作价值，这会导致 Q 值的过高估计，使得智能体对某些动作的价值过于乐观，影响学习的稳定性和准确性。



在强化学习中，Q-learning 是一种经典的值迭代方法，用于求解最优策略。在 Deep Q-Network（DQN）中，引入了深度神经网络作为 Q 函数的逼近器，成功应用于 Atari 游戏等高维状态空间任务。

然而，DQN 存在一个显著的问题：**Q 值的过估计**。这种高估来源于使用同一个网络选择动作并评估该动作的价值，即：

$$
Q(s, a) = r + \gamma \max_{a'} Q(s', a'; \theta)
$$

这种形式容易引发误导性的梯度方向，导致训练不稳定甚至发散。





## Double DQN 的核心思想

Double DQN 的核心是：**将动作选择和动作评估分离**。

具体做法是使用两个网络：

- 当前 Q 网络（在线网络）：$Q(s, a; \theta)$
- 目标 Q 网络（延迟更新的副本）：$Q(s, a; \theta^{-})$

更新目标时，动作由当前 Q 网络选择，Q 值由目标 Q 网络评估：

$$
y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta), \theta^{-})
$$

这样避免了 max 操作与值评估出自同一网络所造成的高估偏差。

## Double DQN 的算法流程

### 初始化：

- 初始化 Q 网络参数$\theta$
- 初始化目标 Q 网络参数$\theta^{-} = \theta$
- 初始化经验回放池 D

### 每一轮训练：

1. 从环境中获得状态$s$，用$\epsilon$-greedy策略选择动作$a$

2. 执行动作$a$ ，观察奖励 $r$ 和新状态$s'$

3. 将$(s, a, r, s')$存入经验回放池 D

4. 从 D 中采样一个 batch 的样本

5. 对每个样本计算 Double DQN 目标：  
   
   $$
   y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta), \theta^{-})
   $$

6. 用均方误差损失：
   
   $$
   L = \left( y - Q(s, a; \theta) \right)^2
   $$
   
   反向传播并更新参数$\theta$

7. 每隔 C 步更新目标网络：
   $\theta^{-} \leftarrow \theta$

## Double DQN 与 DQN 的对比

| 方面    | DQN                                              | Double DQN                                                          |
| ----- | ------------------------------------------------ | ------------------------------------------------------------------- |
| Q 值估计 | 使用同一网络选择和评估动作                                    | 使用两个网络分离动作选择和评估                                                     |
| 是否高估  | 容易高估                                             | 显著缓解高估                                                              |
| 更新目标  | $y = r + \gamma \max_{a'} Q(s', a'; \theta^{-})$ | $y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta), \theta^{-})$ |
| 稳定性   | 相对较差                                             | 更稳定                                                                 |

## 实现参考（伪代码）

```python
# s, a, r, s_prime: 当前状态、动作、奖励、下一个状态
# q_net: 当前Q网络，target_net: 目标网络
# gamma: 折扣因子
# optimizer: 优化器

# 从经验池采样一个 batch
batch = replay_buffer.sample(batch_size)

# 获取下一个状态的最大动作（由当前网络选择）
next_actions = q_net(s_prime).argmax(dim=1)

# 使用目标网络评估这些动作的 Q 值
target_q_values = target_net(s_prime).gather(1, next_actions.unsqueeze(1)).squeeze(1)

# 计算目标 Q 值
targets = rewards + gamma * target_q_values * (1 - dones)

# 当前 Q 网络输出
q_values = q_net(s).gather(1, actions.unsqueeze(1)).squeeze(1)

# 计算损失并反向传播
loss = F.mse_loss(q_values, targets.detach())
optimizer.zero_grad()
loss.backward()
optimizer.step()
```
