[[Temporal-Difference]]
[[Value based RL]]
[强化学习(七)--Q-Learning和Sarsa - 知乎](https://zhuanlan.zhihu.com/p/46850008)
[DQN 从入门到放弃4 动态规划与Q-Learning - 知乎](https://zhuanlan.zhihu.com/p/21378532)(非常好的ref，作者来自kimi，文章日期2016年)

Q-learning是一种**无模型（model-free）**的强化学习算法，用于通过试错学习最优策略。其核心是构建一个**Q函数**（状态-动作值函数），表示在状态$s$下选择动作$a$后能获得的预期累积回报。以下详细介绍其原理、公式及关键步骤。



## 算法概述

Q-Learning是强化学习领域中的一种经典时序差分(Temporal Difference, TD)学习算法，由Christopher Watkins在1989年提出。它是一种off-policy算法，能够直接学习最优策略，而不依赖于所采取的行动策略。Q-Learning通过不断更新状态-动作对的价值估计(Q值)，逐步学习环境中的最优决策。

Q-Learning的核心优势在于它的离线策略特性，使其能够在探索环境的同时，学习可能未经探索的最优行为。这种特性使Q-Learning成为强化学习领域中最流行的算法之一。



## 数学基础

### Markov Decision Process

Q-Learning算法建立在马尔可夫决策过程(Markov Decision Process, MDP)的基础上。MDP是一个五元组$(S, A, P, R, \gamma)$，其中：

- $S$：有限状态集合
- $A$：有限动作集合
- $P$：状态转移概率函数，$P(s'|s,a)$表示在状态$s$下执行动作$a$后转移到状态$s'$的概率
- $R$：奖励函数，$R(s,a,s')$表示在状态$s$下执行动作$a$并转移到状态$s'$时获得的即时奖励
- $\gamma$：折扣因子，$\gamma \in [0,1]$，用于平衡即时奖励与未来奖励



## 核心概念
- **状态（$s$）**：智能体所处的环境状态。
- **动作（$a$）**：智能体可执行的操作。
- **奖励（$r$）**：执行动作后环境返回的即时反馈。
- **折扣因子（$\gamma$）**：权衡当前奖励与未来奖励的重要性，取值范围$[0, 1]$。
- **学习率（$\alpha$）**：控制Q值更新步长，取值范围$[0, 1]$。


## Q-Learning与Bellman Equation
Q函数的目标是最大化**累积折扣奖励**：
$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$
Q值更新基于**贝尔曼方程**，将当前Q值与下一状态的最优Q值结合：
$$
Q(s, a) = \mathbb{E}\left[ r + \gamma \max_{a'} Q(s', a') \mid s, a \right]
$$

其中$s'$是执行动作$a$后的新状态，$a'$ 是 $s'$ 下可能的动作。



## Q-learning更新规则
通过**时序差分（TD）**方法逐步更新Q值：
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]
$$

- **目标值**：$r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a)$
- **TD误差**：目标值与当前Q值的差，乘以学习率$\alpha$调整更新幅度。



### 算法步骤
1. **初始化**：Q表设为任意值（如全零）。
2. **循环每个episode**：
   - 初始化状态$s$。
   - 重复直到终止：
     - 根据策略（如$\epsilon$-greedy）选择动作$a$。
     - 执行$a$，观察奖励$r$和新状态$s'$。
     - 更新Q值： 
       $$ Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right) $$
     - 更新状态：$s \leftarrow s'$

#### 伪代码

```
初始化 Q(s,a) 为任意值
对于每个回合:
    初始化状态 s
    重复(对于回合中的每一步):
        基于当前Q值使用策略选择动作a (如ε-贪婪)
        执行动作a，观察奖励r和新状态s'
        Q(s,a) ← Q(s,a) + α[r + γ * max_a' Q(s',a') - Q(s,a)]
        s ← s'
    直到s是终止状态
```

### 收敛性

理论上，Q-learning需满足：
- 所有状态-动作对被无限次访问。
- 学习率$\alpha$需逐步衰减（如$\alpha_t = \frac{1}{t}$）。
  

此时Q表收敛到**最优Q函数**，导出最优策略$\pi^*(s) = \arg\max_{a} Q(s, a)$。

### Exploration and Exploitation

Q-Learning算法中通常使用$\epsilon$-greedy策略来平衡探索与利用：

$\pi(a|s) = \begin{cases} 1-\epsilon+\frac{\epsilon}{|A(s)|}, & \text{如果 } a = \arg\max_{a'} Q(s,a') \ \frac{\epsilon}{|A(s)|}, & \text{否则} \end{cases}$

其中$|A(s)|$是状态$s$下可选动作的数量，$\epsilon$是一个小的正数，表示探索的概率。通常$\epsilon$会随着训练的进行而逐渐减小，从而从初始的大量探索转向后期的主要利用。

### 与SARSA的比较

Q-Learning和SARSA都是基于时序差分学习的强化学习算法，但它们有几个关键的区别：

1. **策略类型**：
   - Q-Learning是off-policy算法，它学习的是独立于当前所采取策略的最优策略
   - SARSA是on-policy算法，它学习的是基于当前所采取策略的动作价值函数
2. **更新规则**：
   - Q-Learning：$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$
   - SARSA：$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma Q(s',a') - Q(s,a)]$
3. **行为特性**：
   - Q-Learning更为"激进"，它假设未来总是选择最优动作，不考虑探索
   - SARSA更为"保守"，它考虑到实际采取的策略(包括探索)可能带来的风险

### 示例

假设一个状态$s_t$，动作$a_t$获得奖励$r=1$，转移到$s_{t+1}$，且$\max_{a'} Q(s_{t+1}, a')=2$，$\alpha=0.1$，$\gamma=0.9$：
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + 0.1 \times [1 + 0.9 \times 2 - Q(s_t, a_t)]
$$


## 特点

- **Off-policy**：更新时采用最大Q值（非实际执行的动作）。
- **无模型**：无需环境转移概率。
- **适用性**：适用于离散状态-动作空间，大规模问题需结合函数逼近（如DQN）。

Q-learning为强化学习的基础算法，理解其原理是掌握深度Q网络（DQN）等扩展方法的关键。



## 变体和改进版本

Q-Learning 的更新遵循固定的 **核心公式**，即：
$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$
但在实际应用中，有很多 **变体和改进版本** 可以优化其性能，具体更新方式可能有所不同。这些改进通常是为了解决样本效率低、震荡或收敛缓慢等问题。



### Double Q-Learning

**问题**: 标准 Q-Learning 在估计最大 Q 值时可能导致 **过估计偏差**。
 **解决方案**: 使用两组独立的 Q 值表来降低偏差：
$$
Q_1(s, a) \leftarrow Q_1(s, a) + \alpha \left[ R + \gamma Q_2(s', \arg\max_{a'} Q_1(s', a')) - Q_1(s, a) \right]
$$
两组表交替更新，从而避免最大操作引入的系统性偏差。



### SARSA (On-Policy Q-Learning)

**问题**: 标准 Q-Learning 是离策略（off-policy），直接优化期望策略。
 **解决方案**: SARSA 根据智能体实际选择的动作进行更新：
$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma Q(s', a') - Q(s, a) \right]
$$
与 Q-Learning 的区别是使用当前策略选择的动作 $a'$ 而非最大 Q 值。



### DQN (Deep Q-Network)

**问题**: Q-Learning 无法处理连续、复杂状态空间。
 **解决方案**: 使用深度神经网络近似 $Q(s, a)$，并加入经验回放与固定目标网络来稳定训练：
$$
Q_\theta(s, a) \approx Q^*(s, a)
$$



### Q-Learning with Eligibility Traces

**问题**: Q-Learning 仅利用即时误差进行更新，导致收敛速度较慢。
 **解决方案**: 引入资格迹（Eligibility Traces），对多个历史状态进行权重衰减更新：
$$
Q(s, a) \leftarrow Q(s, a) + \alpha \delta e(s, a)
$$


其中 $e(s, a)$ 是资格迹，$\delta$ 是 TD 误差。



### Soft Q-Learning

**问题**: 纯粹的贪婪策略可能导致探索不足。
 **解决方案**: 将动作选择概率通过熵正则化控制：
$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma \sum_{a'} \pi(a'|s') Q(s', a') - Q(s, a) \right]
$$
其中 $\pi(a'|s')$ 是基于策略的概率分布。

## 应用场景

Q-Learning及其变体广泛应用于各种强化学习场景，包括：

1. **游戏AI**：从简单的网格世界游戏到复杂的象棋、围棋、电子游戏
2. **机器人控制**：机器人导航、操作任务、步态优化
3. **推荐系统**：个性化内容推荐、广告投放
4. **资源管理**：网络路由、电力调度、计算资源分配
5. **自动驾驶**：路径规划、驾驶策略学习

## 算法优缺点

### 优点

1. **离线策略**：可以学习最优策略，而不依赖于采取的行动策略
2. **简单直观**：算法简单，容易实现和理解
3. **收敛性保证**：在适当条件下，可以证明算法收敛到最优策略

### 缺点

1. **样本效率**：可能需要大量样本才能学到好的策略
2. **函数近似挑战**：当与函数近似结合时(如深度学习)，可能面临不稳定性和收敛问题
3. **超参数敏感**：性能对学习率、折扣因子等超参数的选择敏感
4. **维度灾难**：在大规模状态空间中，表格形式的Q-Learning变得不切实际

## 实现注意事项

1. **学习率设置**：太大可能导致不稳定，太小会导致学习缓慢；考虑使用学习率衰减策略
2. **探索策略选择**：根据问题特性选择合适的探索策略，如$\epsilon$-贪婪、玻尔兹曼探索
3. **奖励设计**：奖励函数的设计对算法性能有重大影响；避免奖励稀疏和延迟奖励问题
4. **状态表示**：选择有效的状态表示方法，适当的特征工程可以大幅提升性能
5. **经验回放**：对于复杂问题，考虑使用经验回放提高样本利用效率

## 算法扩展与前沿发展

1. **优先级经验回放(Prioritized Experience Replay)**：根据TD误差大小优先回放重要经验
2. **分布式Q-Learning(Distributional Q-Learning)**：学习奖励分布而非期望值
3. **噪声网络(Noisy Networks)**：通过参数空间中的噪声实现更高效的探索
4. **量子Q-Learning**：利用量子计算加速强化学习过程
5. **多智能体Q-Learning**：扩展到多智能体环境，考虑智能体间的相互作用

