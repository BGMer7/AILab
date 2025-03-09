[[Temporal-Difference]]
## SARSA算法详细介绍

### 算法概述

SARSA (State-Action-Reward-State-Action) 是强化学习领域中的一种重要的时序差分 (Temporal Difference, TD) 学习算法。它是一种在线策略 (On-policy) 学习方法，通过在实际使用的策略下采样经验来更新 Q 值函数。

SARSA 算法的名称来源于其更新过程中使用的五元组：当前状态 $s$、当前动作 $a$、获得的奖励 $r$、下一个状态 $s'$ 和在下一个状态选择的动作 $a'$。

### 数学基础

#### MDP框架

SARSA 算法基于马尔可夫决策过程 (Markov Decision Process, MDP)，其中 MDP 由五元组 $(S, A, P, R, \gamma)$ 定义：

- $S$：状态空间
- $A$：动作空间
- $P$：状态转移概率函数，$P(s'|s,a)$ 表示在状态 $s$ 下执行动作 $a$ 后转移到状态 $s'$ 的概率
- $R$：奖励函数，$R(s,a,s')$ 表示从状态 $s$ 执行动作 $a$ 并转移到状态 $s'$ 时获得的奖励
- $\gamma$：折扣因子，$\gamma \in [0,1]$，用于平衡即时奖励与未来奖励的重要性

#### 动作价值函数

SARSA 算法基于动作价值函数 $Q(s,a)$，它表示在状态 $s$ 下采取动作 $a$ 并之后遵循策略 $\pi$ 所能获得的期望累积折扣奖励：

$$
Q^\pi(s,a) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s, A_t = a \right]
$$

其中 $\pi$ 是一个策略，它将状态映射到动作的概率分布：$\pi(a|s)$。

### SARSA 算法

#### 更新规则

SARSA 算法通过迭代的方式更新 $Q(s,a)$ 值，每次迭代基于一个 $(s, a, r, s', a')$ 元组进行更新。其更新规则如下：

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma Q(s',a') - Q(s,a) \right]
$$

其中：
- $\alpha$ 是学习率，$\alpha \in (0,1]$
- $r + \gamma Q(s',a') - Q(s,a)$ 是 TD 误差，表示估计值和实际值之间的差距

#### 完整算法

SARSA 算法的完整过程如下：

1. 初始化 $Q(s,a)$ 为任意值，通常为零
2. 对于每个回合 (episode)：
   - 初始化状态 $s$
   - 使用派生自 $Q$ 的策略（如 $\epsilon$-贪婪策略）选择动作 $a$
   - 对于回合中的每一步，直至到达终止状态：
     - 执行动作 $a$，观察奖励 $r$ 和新状态 $s'$
     - 使用同样的策略选择新动作 $a'$
     - 更新 $Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma Q(s',a') - Q(s,a)]$
     - $s \leftarrow s'$
     - $a \leftarrow a'$

#### 伪代码

```
初始化 Q(s,a) 为任意值
对于每个回合:
    初始化 s
    基于 Q 使用策略（如ε-贪婪）选择 a
    重复直到 s 是终止状态:
        执行动作 a
        观察奖励 r 和新状态 s'
        基于 Q 使用策略选择 a'
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        s ← s'
        a ← a'
```

### 探索与利用策略

在强化学习中，探索(exploration)与利用(exploitation)的权衡是一个重要问题。SARSA 算法通常采用 $\epsilon$-greedy策略来平衡探索与利用：

$$
\pi(a|s) = 
\begin{cases} 
1-\epsilon+\frac{\epsilon}{|A(s)|}, & \text{如果 } a = \arg\max_{a'} Q(s,a') \\
\frac{\epsilon}{|A(s)|}, & \text{否则}
\end{cases}
$$

其中 $|A(s)|$ 是状态 $s$ 下可选动作的数量，$\epsilon$ 是一个小的正数，表示探索的概率。

### 与 Q-learning 的比较

SARSA 与 Q-learning 是两种常见的 TD 学习算法，它们的主要区别在于：

1. **策略类型**：
   - SARSA 是在线策略 (on-policy) 算法，更新值函数时使用的是实际遵循的策略。
   - Q-learning 是离线策略 (off-policy) 算法，更新值函数时使用的是贪婪策略，而不必考虑实际遵循的策略。

2. **更新规则**：
   - SARSA：$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma Q(s',a') - Q(s,a)]$
   - Q-learning：$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$

3. **行为特性**：
   - SARSA 更保守，考虑探索过程中可能遇到的风险。
   - Q-learning 更激进，假设未来可以采取最优动作。

### SARSA(λ) 算法

SARSA 算法的一个重要扩展是 SARSA(λ)，它结合了 TD 学习和蒙特卡洛方法的优点，通过引入资格迹 (eligibility traces) 实现更有效的学习。

#### 资格迹

资格迹为每个状态-动作对 $(s,a)$ 维护一个数值 $e(s,a)$，表示该状态-动作对的"资格"程度，即它对当前更新的贡献度。

$$
e(s,a) = 
\begin{cases} 
\gamma \lambda e(s,a) + 1, & \text{如果 } s=s_t \text{ 且 } a=a_t \\
\gamma \lambda e(s,a), & \text{否则}
\end{cases}
$$

其中 $\lambda \in [0,1]$ 是资格迹衰减参数，控制着对过去经验的重视程度。

#### SARSA(λ) 更新规则

对于每一步，TD 误差为：

$\delta_t = r_{t+1} + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)$

然后，对所有状态-动作对 $(s,a)$ 更新 Q 值：

$Q(s,a) \leftarrow Q(s,a) + \alpha \delta_t e(s,a)$

### 应用场景

SARSA 算法广泛应用于各种强化学习问题，特别是：

1. **机器人导航**：在不确定环境中寻找最优路径。
2. **游戏 AI**：如围棋、国际象棋等游戏中的智能体学习。
3. **自动驾驶**：学习适应不同的道路条件和交通情况。
4. **资源分配**：优化计算资源或物流系统中的决策。

### 算法优缺点

#### 优点

1. **在线学习**：可以在交互过程中不断学习和改进。
2. **策略评估和改进**：同时进行策略评估和策略改进。
3. **安全性**：考虑探索过程中的风险，更适合需要安全保障的场景。

#### 缺点

1. **样本效率**：可能需要大量样本才能学到好的策略。
2. **参数敏感**：性能受到学习率、折扣因子等参数的影响。
3. **收敛速度**：在某些复杂环境下收敛速度可能较慢。

### 实现注意事项

1. **学习率衰减**：随着学习的进行，逐渐减小学习率 $\alpha$ 可以提高算法的稳定性。
2. **探索率衰减**：随着学习的进行，逐渐减小探索率 $\epsilon$ 可以从探索转向利用。
3. **奖励设计**：合理设计奖励函数对算法性能有重要影响。
4. **状态表示**：选择合适的状态表示方法对减少状态空间、提高学习效率非常重要。

### 算法扩展

除了 SARSA(λ)，SARSA 算法还有其他一些扩展：

1. **期望 SARSA (Expected SARSA)**：使用下一个状态下所有可能动作的期望值进行更新：
   $Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \sum_{a'} \pi(a'|s') Q(s',a') - Q(s,a)]$

2. **深度 SARSA (Deep SARSA)**：使用深度神经网络近似 Q 函数，适用于高维或连续的状态空间。

3. **多步 SARSA (n-step SARSA)**：考虑未来多步的奖励来更新当前的 Q 值：
   $Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha [r_{t+1} + \gamma r_{t+2} + ... + \gamma^{n-1} r_{t+n} + \gamma^n Q(s_{t+n},a_{t+n}) - Q(s_t,a_t)]$
