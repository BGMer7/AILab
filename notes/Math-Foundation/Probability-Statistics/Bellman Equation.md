# 贝尔曼方程

贝尔曼方程（Bellman Equation）是强化学习和动态规划中的核心工具，用于递归地定义状态的价值函数（Value Function）。它基于马尔可夫性质，将当前状态的价值表示为未来状态的价值和即时奖励的组合。



## 价值函数（Value Function）

价值函数表示在某个状态下，智能体可以期望获得的累积奖励。对于任意状态 $s$，其价值函数为：

$$
V(s) = \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t R(s_t) \mid s_0 = s \right]
$$

其中：
- $V(s)$：状态 $s$ 的价值。
- $\gamma$：折扣因子，$0 \leq \gamma \leq 1$。
- $R(s_t)$：时间步 $t$ 的即时奖励。
- $\mathbb{E}$：期望运算。



## 贝尔曼期望方程

贝尔曼期望方程利用马尔可夫性质，将价值函数递归地定义为即时奖励和下一状态价值的加权和：

$$
V(s) = R(s) + \gamma \sum_{s'} P(s' \mid s) V(s')
$$

其中：
- $R(s)$：当前状态 $s$ 的即时奖励。
- $P(s' \mid s)$：从状态 $s$ 转移到状态 $s'$ 的概率。
- $\gamma V(s')$：折扣后的下一状态价值。



## 贝尔曼最优方程

在强化学习中，策略优化的目标是找到最优策略，使得价值函数最大化。最优价值函数满足以下递归关系：

$$
V^*(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^*(s') \right]
$$

其中：
- $V^*(s)$：状态 $s$ 的最优价值。
- $a$：当前选择的动作。
- $R(s, a)$：在状态 $s$ 下执行动作 $a$ 的即时奖励。
- $P(s' \mid s, a)$：从状态 $s$ 执行动作 $a$ 后转移到状态 $s'$ 的概率。



## 动作价值函数（Q函数）

动作价值函数表示在状态 $s$ 执行动作 $a$ 时，智能体可以期望获得的累积奖励。其贝尔曼方程为：

$$
Q(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q(s', a')
$$

**Q函数和价值函数的关系**：
- 最优价值函数可以通过最优Q函数得到：
  $$
  V^*(s) = \max_a Q^*(s, a)
  $$



## 贝尔曼方程的重要性

1. **递归定义价值函数**：贝尔曼方程提供了分解问题的方式，使得复杂问题可以通过子问题递归求解。
2. **策略优化基础**：强化学习中的策略迭代和价值迭代方法都基于贝尔曼方程。
3. **广泛应用**：在动态规划、路径规划、游戏AI等领域中具有核心地位。



## 贝尔曼最优方程与动作价值函数的关系

贝尔曼最优方程描述了在最优策略下，状态的最大累积回报；而动作价值函数（Q函数）则进一步细化，描述了在给定状态 $s$ 和动作 $a$ 下的累积回报。

两者之间的关系可以通过最优Q函数来表示状态的最优价值。



### 贝尔曼最优方程

最优价值函数 $V^*(s)$ 的定义是：

$$
V^*(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^*(s') \right]
$$


### 动作价值函数（Q函数）

最优动作价值函数 $Q^*(s, a)$ 描述了在状态 $s$ 下选择动作 $a$ 所能获得的最大累积奖励：

$$
Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q^*(s', a')
$$



### 关系公式

最优价值函数可以通过最优动作价值函数表示为：

$$
V^*(s) = \max_a Q^*(s, a)
$$

这意味着，对于任意状态 $s$，其最优价值等于在所有可能动作 $a$ 中，选择使 $Q^*(s, a)$ 最大的动作的值。

同样，最优动作价值函数也与最优价值函数相联系：

$$
Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^*(s')
$$

这说明，$Q^*(s, a)$ 是在状态 $s$ 执行动作 $a$ 后获得的即时奖励 $R(s, a)$ 加上后续状态 $s'$ 的最优价值 $V^*(s')$ 的加权和。



### 直观理解

- **$V^*(s)$**：表示智能体在状态 $s$ 时，如果遵循最优策略所能获得的最大累积奖励。
- **$Q^*(s, a)$**：表示智能体在状态 $s$ 下执行动作 $a$，然后遵循最优策略所能获得的最大累积奖励。

两者的关系可以看作：
- $Q^*(s, a)$ 是更具体的累积回报，因为它包括了当前选择的动作 $a$。
- $V^*(s)$ 是对所有可能动作的全局最优累积回报。



### 应用场景

1. **策略选择**：
   - 通过 $Q^*(s, a)$ 可以找到最优策略：
     $$
     \pi^*(s) = \arg\max_a Q^*(s, a)
     $$

2. **计算最优值**：
   - $V^*(s)$ 提供了某状态的最大长期收益，是智能体评估状态的重要依据。

3. **强化学习算法**：
   - Q-learning 和 SARSA 等算法直接基于动作价值函数 $Q(s, a)$ 进行学习和更新。


