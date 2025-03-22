[[Q-Learning]]

## DQN相关论文

https://blog.csdn.net/Xixo0628/article/details/120434658

1. **《Playing Atari with Deep Reinforcement Learning》**：这是DQN的开山之作，由Mnih等人于2013年提出。该论文将深度神经网络（DNN）与Q-learning相结合，利用DNN强大的拟合能力来估计动作的Q值。其网络架构采用3层卷积层，因为基于像素计算，所以不需要调整架构就能在各种游戏上取得较好的结果。同时，该论文还引入了replay buffer的技巧，打破了交互序列的相关性，提高了样本的使用效率。
    https://arxiv.org/abs/1312.5602

2. **《Deep Recurrent Q-Learning for Partially Observable MDPs》**：由Hausknecht和Stone于2015年提出。该论文针对部分可观察的马尔可夫决策过程（POMDP），在传统DQN的基础上引入了长短期记忆网络（LSTM），使网络能够“记忆”一些画面，从而在单靠一帧无法获得全部信息的情况下（如小球的运动方向），提供更好的决策支持。
    https://arxiv.org/abs/1507.06527

3. **《Dueling Network Architectures for Deep Reinforcement Learning》**：由Wang等人于2015年提出。该论文提出了dueling network架构，将状态价值评估和动作价值评估“分离”。通过修改DQN最后一层全连接层的输出，将其分为当前状态的价值V和各个动作的价值向量A，使得对V、A的估计更加准确，从而更好地评价当前局面的状态价值和动作价值。
    https://arxiv.org/abs/1511.06581

4. **《Deep Reinforcement Learning with Double Q-learning》**：由Hasselt等人于2015年提出。该论文将选择与评估分离，引入了Double Q-learning的概念。通过将挑选最优动作的资格交给update network，再交给target network进行评估，使得对于“最优”动作的估计是无偏的，从而解决了DQN中的过估计问题。
    https://arxiv.org/abs/1509.06461

5. **《Prioritized Experience Replay》**：由Schaul等人于2015年提出。该论文在抽取经验池中过往经验样本时，采取按优先级抽取的方法。传统随机抽取经验忽略了经验之间的重要程度，而该方法用二八定律的思想，重点学习那百分之二十能够带来百分之八十收益的行为，从而提高学习效率和效果。
    https://arxiv.org/abs/1511.05952

6. **《Rainbow: Combining Improvements in Deep Reinforcement Learning》**：由Hessel等人于2017年提出。该论文是DQN的集大成之作，整合了之前多种不冲突的改进方法，如Double DQN、Prioritized Experience Replay等，实验结果表现出色，训练后的结果在多个方面都有显著提升，且团队还进行了消融实验来验证各方法的有效性。
    https://arxiv.org/abs/1710.02298

## Main Network & Target Network

### 为什么要两个网络？

DQN（Deep Q-Network）需要一个主网络（main network）和一个目标网络（target network）的主要原因是**为了提高训练的稳定性**。

1. **目标值的不稳定性：** 在强化学习中，我们使用时序差分（Temporal Difference, TD）方法来更新Q值。在DQN中，目标Q值的计算通常涉及到当前Q网络本身。具体来说，当我们计算一个状态-动作对`(s, a)`的目标Q值时，我们会使用Q网络来估计下一个状态`s'`下能够获得的最大Q值。如果我们在每次更新主网络后都立即使用更新后的网络来计算目标Q值，那么目标值也会随之快速变化。这种频繁变化的目标值会导致训练过程不稳定，甚至可能发散。

2. **打破目标值与预测值的相关性：** 使用同一个网络来生成目标Q值和预测当前状态-动作对的Q值，会引入很强的相关性。这就像试图用一个不断变化的标准来衡量自己，很难收敛到一个稳定的状态。目标网络的引入，通过延迟更新目标Q值的计算网络，可以有效地打破这种相关性。

3. **提供更稳定的训练目标：** 目标网络是主网络的一个滞后版本，它的参数会定期（例如，每隔一定的步数）从主网络复制过来，但在复制之间保持不变。这样，在一段时间内，计算目标Q值所使用的网络参数是固定的，从而为训练主网络提供了一个更稳定的目标。这有助于减小训练过程中的震荡，使得学习过程更加平稳。

**总结来说，目标网络的作用类似于在训练过程中引入了一个“时间上的缓冲”，使得目标Q值的变化不会过于剧烈，从而提高DQN算法的稳定性和收敛性。** 主网络负责学习和更新Q值函数，而目标网络则提供一个相对稳定的目标来指导主网络的学习。



在DQN中，使用两个网络（main network和target network）的主要原因是为了提高训练的稳定性和减少Q值估计的方差。以下是具体的原因：

1. **减少最大化偏差[[Maximization Bias]]**：
   
   - 在传统的Q-learning中，使用同一个网络来选择和评估动作可能会导致最大化偏差。这是因为网络可能会倾向于高估某些动作的价值，从而导致策略的不稳定。
   - 通过使用两个不同的网络，main network用于选择动作，而target network用于评估这些动作的价值，可以减少这种偏差。target network的参数是固定的或更新较慢，从而提供了一个更稳定的评估基准。

2. **提高训练稳定性**：
   
   - 如果只使用一个网络来同时选择和评估动作，那么在每次更新时，网络的参数会发生变化，这可能导致目标值（target values）的剧烈波动，从而使得训练过程变得不稳定。
   - 使用两个网络可以使得目标值相对稳定。target network的参数更新频率较低，或者在一定时间间隔后从main network复制过来，这样可以避免目标值的频繁变化，提高训练的稳定性。

3. **减少相关性**：
   
   - 在强化学习中，连续的样本之间往往具有高度的相关性。如果使用同一个网络来处理这些相关样本，可能会导致模型过拟合到这些相关性中，从而影响泛化能力。
   - 使用两个网络可以减少这种相关性的影响。main network用于当前策略的探索和更新，而target network用于生成目标值，从而在一定程度上减少了连续样本之间的相关性。

4. **简化优化过程**：
   
   - 使用两个网络可以简化优化过程。main network负责根据当前策略选择动作，而target network负责根据旧的策略评估这些动作的价值。这种分离使得优化过程更加清晰和高效。

总结来说，使用main network和target network的主要目的是为了减少最大化偏差、提高训练稳定性、减少样本相关性的影响，并简化优化过程。这种设计使得DQN在处理复杂的强化学习任务时更加有效和稳定。





# DQN

DQN（Deep Q-Network）算法是一种结合了深度学习和强化学习中Q-Learning算法的深度强化学习方法，由DeepMind团队于2015年首次提出。它通过使用深度神经网络来近似Q值函数，解决了传统Q-Learning算法在处理高维状态空间和大规模动作空间时的困难，使得强化学习能够应用于更复杂的任务，如玩电子游戏、机器人控制等。

## Q-Learning基础

Q-Learning是一种基于值函数的强化学习算法，其核心思想是学习一个Q值表，记录每个状态-动作对的价值。Q值更新公式为：
$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t)]
$$
其中，$\alpha$是学习率，$\gamma$是折扣因子。

### DQN的改进

DQN将Q-Learning中的Q值表替换为深度神经网络，网络的输入是状态，输出是对应于各个可能动作的Q值。为了提高学习的稳定性和效率，DQN引入了以下关键技术：

1. **经验回放（[[Experience Replay]]）**：
   将智能体与环境交互产生的经验（状态、动作、奖励、新状态）存储在一个数据集中，然后从中随机抽取样本进行学习，以打破数据之间的相关性。

2. **目标网络（Target Network）**：
   使用两个神经网络，一个是在线网络（Q网络），用于选择动作；另一个是目标网络，用于计算TD目标（Temporal-Difference Target）。目标网络的参数定期从在线网络复制过来，这样可以避免在线网络的快速更新导致的震荡问题。

### Target Network

在DQN（Deep Q-Network）算法中，目标网络（Target Network）是一个关键的组件，它用于提高学习的稳定性和收敛速度。目标网络的基本思想是使用两个神经网络：一个称为在线网络（Online Network），另一个称为目标网络。在线网络用于根据当前状态选择动作，而目标网络用于计算Q值更新公式中的目标值（TD目标）。目标网络的参数定期从在线网络复制过来，而不是在每一步都进行更新。这种设计有助于减少学习过程中的波动和不稳定。

#### 目标网络的作用

1. **稳定学习过程**：通过使用固定的目标网络来计算目标Q值，可以避免在线网络快速更新导致的Q值估计不稳定，从而提高学习的稳定性。
2. **减少训练波动**：目标网络的参数不是每次更新都改变，这有助于减少训练过程中的波动，使学习过程更加平滑。
3. **提高收敛速度**：通过使用目标网络，DQN算法能够更快地收敛到最优策略。

#### 目标网络的实现步骤

1. **初始化两个网络**：在线网络和目标网络具有相同的网络结构，但参数不同。在线网络用于学习和更新，目标网络用于提供稳定的目标值。
2. **定期更新目标网络**：每隔一定的步数（如每C步），将在线网络的参数复制到目标网络。这可以通过设置一个计数器来实现，当计数器达到预设值时，执行参数复制操作。
3. **计算目标Q值**：在更新在线网络时，使用目标网络来计算目标Q值，即下一个状态的最大Q值。
4. **更新在线网络**：通过最小化在线网络预测的Q值和目标网络计算的目标Q值之间的差距来更新在线网络的参数。

#### 目标网络的代码示例

以下是一个使用PyTorch实现目标网络的简单示例：

```python
# 初始化在线网络和目标网络
online_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(online_net.state_dict())  # 复制在线网络的参数到目标网络

# 定义优化器
optimizer = optim.Adam(online_net.parameters())

# 训练循环
for episode in range(num_episodes):
    state = env.reset()
    for step in range(max_steps):
        # 选择动作
        action = select_action(online_net, state, epsilon)

        # 执行动作并获取下一个状态和奖励
        next_state, reward, done, _ = env.step(action)

        # 存储经验到经验回放缓冲区
        replay_buffer.append((state, action, reward, next_state, done))

        # 更新状态
        state = next_state

        # 如果缓冲区有足够的样本，进行学习
        if len(replay_buffer) >= batch_size:
            # 从缓冲区中随机采样一批数据
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # 转换为张量
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

            # 计算当前Q值
            current_q = online_net(states).gather(1, actions)

            # 计算目标Q值
            with torch.no_grad():
                next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                target_q = rewards + gamma * next_q * (1 - dones)

            # 计算损失
            loss = nn.MSELoss()(current_q, target_q)

            # 反向传播并优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 定期更新目标网络
        if step % update_target_every == 0:
            target_net.load_state_dict(online_net.state_dict())

        if done:
            break
```

#### 目标网络的改进

目标网络的基本形式在DQN算法中已经表现出色，但研究人员也在不断探索其改进方法。例如，Double DQN算法通过使用在线网络来选择动作，而目标网络来评估动作的价值，从而减少过高的Q值估计。此外，还有一些研究提出了软更新目标网络的方法，即不是每隔一定步数完全复制参数，而是逐步更新目标网络的参数，以进一步提高学习的稳定性。

目标网络是DQN算法中的一个重要组件，它通过提供稳定的目标值来提高学习的稳定性和收敛速度。通过定期从在线网络复制参数，目标网络能够在学习过程中保持目标值的稳定性，从而帮助智能体更有效地学习最优策略。

### Q值更新公式

在DQN中，Q值的更新公式为：
$$
Q(s, a) = r + \gamma \max_{a'} Q_{\text{target}}(s', a')
$$
其中，$Q_{\text{target}}(s', a')$是通过目标网络计算出的Q值，而$Q(s, a)$则是通过在线网络计算出的Q值。

## 算法流程

1. **初始化**：
   初始化在线网络和目标网络（结构相同但参数不同），创建经验回放缓冲区。

2. **探索与利用**：
   智能体在每个时间步选择一个动作，可以是随机的（探索）或根据在线网络预测的Q值选择的（利用），通常使用ε-greedy策略。

3. **交互与存储**：
   智能体根据选择的动作与环境交互，观察新的状态和奖励，并将转移（状态，动作，奖励，新状态）存储在经验回放缓冲区中。

4. **学习**：
   从经验回放缓冲区中随机抽取一批样本，计算每个样本的目标值（$r + \gamma \max_{a'} Q_{\text{target}}(s', a')$），通过最小化网络预测的Q值和目标值之间的差距来更新在线网络的参数。

5. **更新目标网络**：
   每隔一定的步数，将在线网络的参数复制到目标网络。

6. **迭代**：
   重复上述步骤，直到满足停止条件（如达到最大步数或达到预定的性能标准）。

## 代码实现示例

以下是一个使用PyTorch实现的简单的DQN算法的例子，假设环境是OpenAI Gym的CartPole-1环境：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque
import random

# 定义Q网络
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        return self.fc(state)
# 创建环境
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
print(state_dim, action_dim)
# 创建网络
online_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(online_net.state_dict())
# 创建优化器
optimizer = optim.Adam(online_net.parameters())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
online_net.to(device)
target_net.to(device)
# 创建经验回放缓冲区
replay_buffer = deque(maxlen=10000)
# 设置超参数
epsilon = 1.0  # 探索率
epsilon_decay = 0.995  # 探索率衰减
min_epsilon = 0.01  # 最小探索率
gamma = 0.99  # 折扣因子
batch_size = 64  # 批大小
update_target_every = 100  # 更新目标网络的频率
max_steps = 10000  # 最大步数
# 训练过程
for step in range(max_steps):
    # 选择动作
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    # epsilon-greedy策略
    if np.random.rand() < epsilon:
        action = env.action_space.sample()  # 探索
    else:
        with torch.no_grad():
            action = torch.argmax(online_net(state)).item()  # 利用

    # 执行动作并存储转移
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated  # 合并终止和截断条件
    next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
    reward = torch.tensor([reward], dtype=torch.float32).to(device)
    replay_buffer.append((state, action, reward, next_state, done))
    state = next_state

    # 学习
    if len(replay_buffer) >= batch_size:
        minibatch = random.sample(replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.cat(states).to(device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.cat(rewards).to(device)
        next_states = torch.cat(next_states).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        q_values = online_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q_values = target_net(next_states).max(1)[0]
            target_q_values = rewards + gamma * (1 - dones) * max_next_q_values

        loss = nn.functional.mse_loss(q_values, target_q_values.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新目标网络
        if step % update_target_every == 0:
            target_net.load_state_dict(online_net.state_dict())

    # 更新探索率
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # 检查是否完成
    if done:
        break
```

## 总结

DQN算法通过结合深度学习和强化学习，解决了传统Q-Learning算法在高维状态空间中的应用难题。其引入的经验回放和目标网络技术，有效提高了学习的稳定性和效率。通过上述代码示例，您可以更直观地理解DQN算法的实现过程，并将其应用于实际的强化学习任务中。