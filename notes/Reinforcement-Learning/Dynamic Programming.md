Dynamic Programming（动态规划, DP）是强化学习中一种基于 **模型（Model-based）** 的求解方法，其前提是已知环境的 **状态转移概率** 和 **即时奖励函数**。DP 通常用于求解 **马尔可夫决策过程（Markov Decision Process, MDP）** 的最优策略与最优值函数。

## 基本概念

MDP 定义为一个五元组： $S, A, P, R, \gamma$

- $S$：有限状态空间
- $A$：有限动作空间
- $P(s'|s, a)$：状态转移概率
- $R(s, a)$：即时奖励函数
- $\gamma$：折扣因子

强化学习中的主要目标是求解 **最优策略 $\pi^*$，使得累积折扣回报最大化**：
$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

### [[Math-Foundation/Probability-Statistics/Bellman Equation]]

DP 的核心思想依赖于 **贝尔曼方程** 来递归定义值函数和优化策略。

- **状态值函数 $V^{\pi}(s)$：**

$$
V^{\pi}(s) = \mathbb{E}_{\pi} \left[ G_t \mid s_t = s \right] = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s, a) \left[ R(s, a) + \gamma V^{\pi}(s') \right]
$$

- **状态-动作值函数 $Q^{\pi}(s, a)$：**

$$
Q^{\pi}(s, a) = \mathbb{E}_{\pi} \left[ G_t \mid s_t = s, a_t = a \right] = \sum_{s'} P(s'|s, a) \left[ R(s, a) + \gamma \sum_{a'} \pi(a'|s') Q^{\pi}(s', a') \right]
$$



## 动态规划算法

### 策略评估（Policy Evaluation）

策略评估用于计算给定策略 $\pi$ 的值函数 $V^{\pi}(s)$。通常通过 **贝尔曼期望方程** 反复迭代来逼近真实值。
$$
V_{k+1}(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s, a) \left[ R(s, a) + \gamma V_k(s') \right]
$$
**终止条件**：当值函数的变化小于阈值 $\theta$ 时停止。

### 策略改进（Policy Improvement）

策略改进通过基于当前值函数 $V^{\pi}(s)$ 选择新的动作，从而优化策略。
$$
\pi'(s) = \arg\max_a \sum_{s'} P(s'|s, a) \left[ R(s, a) + \gamma V^{\pi}(s') \right]
$$


**策略改进定理**：
 若 $\pi'$ 在每个状态下都比 $\pi$ 好，则 $V^{\pi'}(s) \geq V^{\pi}(s)$。

### 策略迭代（Policy Iteration）

策略迭代通过交替执行 **策略评估** 和 **策略改进** 来逐步逼近最优策略。

**算法步骤：**

1. 初始化任意策略 $\pi_0$
2. 重复以下步骤，直到策略收敛：
   - **策略评估**：计算 $V^{\pi_i}$
   - **策略改进**：更新策略为 $\pi_{i+1}$

### 价值迭代（Value Iteration）

价值迭代是策略迭代的简化版本，将策略评估与策略改进合并为一步。

$$
V_{k+1}(s) = \max_a \sum_{s'} P(s'|s, a) \left[ R(s, a) + \gamma V_k(s') \right]
$$
**终止条件**：当 $|V_{k+1}(s) - V_k(s)| < \theta$ 时停止。



## DP 与 强化学习的关联

- **模型依赖性**：DP 需要环境的转移概率 $P(s'|s, a)$ 和奖励函数 $R(s, a)$，而强化学习通常在未知模型环境中工作。
- **迭代更新**：DP 和 Q-Learning、TD 方法都通过迭代逼近最优值函数和策略。
- **贝尔曼方程的核心地位**：无论是 DP 还是时序差分方法，都基于贝尔曼方程。



## 公式与算法总结

- **贝尔曼最优方程**： $V^*(s) = \max_a \sum_{s'} P(s'|s, a) \left[ R(s, a) + \gamma V^*(s') \right]$
- **值迭代更新公式**： $V_{k+1}(s) = \max_a \sum_{s'} P(s'|s, a) \left[ R(s, a) + \gamma V_k(s') \right]$

DP 提供了强化学习算法的理论基础，是理解和构建价值函数方法的重要工具。



## 实例

```python
import numpy as np
# 定义网格世界
class GridWorld:
    def __init__(self, size=4, goal=(3, 3), reward=1):
        self.size = size
        self.goal = goal
        self.reward = reward
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 右、左、下、上
        self.discount = 0.9

    def is_valid(self, state):
        return 0 <= state[0] < self.size and 0 <= state[1] < self.size

    def step(self, state, action):
        next_state = (state[0] + action[0], state[1] + action[1])
        if not self.is_valid(next_state):
            next_state = state  # 如果超出边界，保持原位
        reward = self.reward if next_state == self.goal else 0
        done = next_state == self.goal
        return next_state, reward, done
# 价值迭代算法
def value_iteration(grid_world, threshold=1e-4):
    # 初始化状态价值函数
    V = np.zeros((grid_world.size, grid_world.size))
    delta = float('inf')

    while delta > threshold:
        delta = 0
        for i in range(grid_world.size):
            for j in range(grid_world.size):
                state = (i, j)
                if state == grid_world.goal:
                    continue  # 终止状态的价值为0，无需更新
                v = V[state]
                # 计算每个动作的期望价值
                action_values = []
                for action in grid_world.actions:
                    next_state, reward, _ = grid_world.step(state, action)
                    action_values.append(reward + grid_world.discount * V[next_state])
                V[state] = max(action_values)
                delta = max(delta, abs(v - V[state]))
    
    # 从价值函数中提取最优策略
    policy = np.zeros((grid_world.size, grid_world.size), dtype=int)
    for i in range(grid_world.size):
        for j in range(grid_world.size):
            state = (i, j)
            if state == grid_world.goal:
                continue
            action_values = []
            for action in grid_world.actions:
                next_state, reward, _ = grid_world.step(state, action)
                action_values.append(reward + grid_world.discount * V[next_state])
            print(i, j, action_values)
            policy[state] = np.argmax(action_values)

    return V, policy

grid_world = GridWorld(goal=(2, 3))
V, policy = value_iteration(grid_world)

print("状态价值函数 V:")
print(V)

print("\n最优策略（动作索引）:")
print(policy)

# 将动作索引转换为可读的动作
action_names = {0: "→", 1: "←", 2: "↓", 3: "↑"}
policy_readable = np.vectorize(action_names.get)(policy)
print("\n最优策略（可读形式）:")
print(policy_readable)

状态价值函数 V:
[[0.6561 0.729  0.81   0.9   ]
 [0.729  0.81   0.9    1.    ]
 [0.81   0.9    1.     0.    ]
 [0.729  0.81   0.9    1.    ]]

最优策略（动作索引）:
[[0 0 0 2]
 [0 0 0 2]
 [0 0 0 0]
 [0 0 0 3]]

最优策略（可读形式）:
[['→' '→' '→' '↓']
 ['→' '→' '→' '↓']
 ['→' '→' '→' '→']
 ['→' '→' '→' '↑']]

```

###  价值迭代 value_iteration

- 初始化状态价值函数V，并设置收敛阈值 threshold 。
- 对每个状态，计算所有可能动作的期望价值，并取最大值更新当前状态的价值。
- 当状态价值函数的变化小于阈值时，认为收敛。
- 从收敛后的价值函数中提取最优策略。

价值迭代是一种**动态规划（DP）**算法，通过迭代更新状态价值函数 $V(s)$，最终收敛到**最优状态价值函数** $V^*(s)$，并从中提取最优策略 $\pi^*$。其核心是**贝尔曼最优方程**：

$$
V_{k+1}(s) = \max_{a} \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s') \right]
$$

#### **算法步骤**
1. **初始化**：所有状态的价值 $V(s)$ 设为0。
2. **迭代更新**：
   
   - 对每个状态 $s$，计算所有可能动作的期望价值 $Q(s,a)$。
   - 将 $V(s)$ 更新为所有动作中的最大Q值：
     $$
     V(s) \leftarrow \max_{a} \left[ R(s,a) + \gamma V(s') \right]
     $$
   - 重复直到 $V(s)$ 的变化小于阈值 $\Delta$。
3. **策略提取**：根据收敛的 $V^*(s)$，选择每个状态下最大化Q值的动作。

#### **代码实现**
- 环境设定为4x4网格，目标状态在 $(2,3)$，奖励为1。
- 每次动作转移是确定的（无随机性），因此 $P(s'|s,a) = 1$。

---

### 最终价值函数的形式
最终的价值函数 $V$ 是一个**4x4矩阵**，每个元素 $V[i,j]$ 表示状态 $(i,j)$ 的最优价值。例如：
- 目标状态 $(2,3)$ 的价值为1（直接获得奖励）。
- 邻近目标的状态（如 $(1,3)$）价值为 $1 \times \gamma^1 = 0.9$，依此类推。

---

### $ V $ 与 $ \pi $ 的更新过程的区别
#### **共同点**
两者均遍历所有状态，并计算每个动作的Q值 $ Q(s,a) = R(s,a) + \gamma V(s') $。

#### **区别**
| 步骤         | 价值函数 $ V $ 更新               | 策略 $ \pi $ 提取                       |
| ------------ | --------------------------------- | --------------------------------------- |
| **目标**     | 找到状态的最优价值 $ V^*(s) $     | 根据 $ V^*(s) $ 选择最优动作            |
| **更新规则** | $ V(s) \leftarrow \max_a Q(s,a) $ | $ \pi(s) \leftarrow \arg\max_a Q(s,a) $ |
| **依赖关系** | 先迭代更新 $ V $，再提取 $ \pi $  | 依赖已收敛的 $ V $                      |



这是一个很好的观察！让我详细解释一下求解最优价值函数(V)和最优策略(policy)的异同。

从代码结构上看，这两部分确实非常相似 - 都是在计算每个状态下所有动作的价值。但它们有着本质的区别：

4. 求解最优价值函数(V)的过程：
```python
while True:
    delta = 0
    for s in range(env.state_count):
        v = V[s]
        action_values = []
        for a in range(env.action_count):
            next_s = env.get_next_state(s, a)
            reward = env.rewards[next_s // env.width, next_s % env.width]
            action_values.append(reward + gamma * V[next_s])
        V[s] = max(action_values)  # 更新价值函数
        delta = max(delta, abs(v - V[s]))
    if delta < theta:
        break  # 当变化很小时停止迭代
```

5. 求解最优策略(policy)的过程：
```python
policy = np.zeros(env.state_count, dtype=int)
for s in range(env.state_count):
    action_values = []
    for a in range(env.action_count):
        next_s = env.get_next_state(s, a)
        reward = env.rewards[next_s // env.width, next_s % env.width]
        action_values.append(reward + gamma * V[next_s])
    policy[s] = np.argmax(action_values)  # 记录最优动作
```

关键区别在于：

6. **迭代性质**：
   - 价值函数求解是一个迭代的过程。因为每个状态的价值依赖于其他状态的价值，我们需要反复迭代直到收敛。
   - 策略提取只需要执行一次，因为此时我们已经有了收敛的价值函数。

7. **存储内容**：
   - 价值函数存储的是每个状态的预期回报值（V[s] = max(action_values)）
   - 策略存储的是每个状态下应该采取的最优动作编号（policy[s] = argmax(action_values)）

8. **用途不同**：
   - 价值函数告诉我们"在每个状态能获得多大的收益"
   - 策略告诉我们"在每个状态应该采取什么动作"

可以用一个具体的例子来说明：
假设在某个状态s，我们计算出四个动作的期望回报分别是：

```python
action_values = [0.8, 0.9, 0.7, 0.6]

# 价值函数会记录最大值
V[s] = max(action_values)  # 存储 0.9

# 策略会记录最大值对应的动作编号
policy[s] = np.argmax(action_values)  # 存储 1（因为0.9在索引1的位置）
```

这种设计反映了强化学习中的一个重要概念：价值函数和策略是相互补充的两个视角。价值函数告诉我们每个状态的"价值"，而策略则告诉我们如何行动才能达到这个价值。你可以把价值函数想象成地图上的海拔高度，而策略就是告诉你往哪个方向走可以到达最高点。





### 最优策略的逻辑
最优策略的目的是**从任意状态出发，选择动作使长期回报最大化**，而非“从数值最大的地方出发”。具体表现为：
- **动作选择**：在每个状态 $s$，策略 $\pi(s)$ 选择使 $Q(s,a)$ 最大的动作 $a$。
- **价值导向**：高价值状态（如靠近目标的状态）会通过 $ \gamma V(s') $ 影响相邻状态的动作选择。

#### **示例分析**
- 在状态 $(0,0)$，最优动作为“→”（右），因为向右移动到 $(0,1)$ 的价值 $V(0,1) = 0.729$ 高于其他方向。
- 在状态 $(3,3)$（目标），策略动作为“↑”（上），但由于该状态是终止状态，实际无需动作（代码中可能未处理终止状态动作）。

---

### 公式与符号总结
| 公式/符号      | 含义                                             |
| -------------- | ------------------------------------------------ |
| $ V(s) $       | 状态 $ s $ 的价值函数                            |
| $ \gamma $     | 折扣因子（代码中为0.9），权衡未来回报的重要性    |
| $ R(s,a) $     | 执行动作 $ a $ 后获得的即时奖励                  |
| $ Q(s,a) $     | 动作价值函数，$ Q(s,a) = R(s,a) + \gamma V(s') $ |
| $ \arg\max_a $ | 选择使表达式最大的动作 $ a $                     |

---

通过价值迭代，智能体逐步修正对状态的估值，最终生成一个全局最优策略。价值函数 $V$ 是策略优化的基础，而策略 $\pi$ 是价值函数的具体行动表现。



### Leetcode 中的DP问题

LeetCode上的许多动态规划问题可以借鉴强化学习中DP（Dynamic Programming）通用方法进行求解，不过需要结合问题的特性做适当调整。

强化学习中的DP方法主要包括 **策略迭代** 和 **价值迭代**，它们与经典的动态规划问题在思想上有一定共通点。以下是一些关键的类比与应用建议：

------

#### 状态表示 (State Representation)

强化学习中的状态一般包括环境的所有信息。LeetCode的DP问题同样需要定义状态，例如：

- **背包问题**：状态可以用 `(当前物品索引, 剩余容量)` 表示。
- **字符串编辑距离**：状态可以用 `(i, j)` 表示当前两个字符串的前缀长度。
- **网格路径问题**：状态可以用 `(x, y)` 表示当前在网格中的位置。

#### 动作选择 (Action Selection)

在强化学习中，动作决定了状态的转移。LeetCode中的“动作”就是影响状态变化的选择：

- **背包问题**：选择是否放入当前物品。
- **路径问题**：选择向右、向下或其他移动方向。
- **博弈问题**：选择下一个玩家的最佳策略。

#### 状态转移方程 (State Transition Equation)

这是 DP 和强化学习的核心：

- 强化学习中的 Bellman 方程：

  $V(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s'|s, a) V(s') \right]$

  - $V(s)$：当前状态的价值
  - $R(s, a)$：执行动作 $a$ 的奖励
  - $P(s'|s, a)$：从状态 $s$ 到 $s'$ 的概率

- LeetCode 动态规划中的状态转移也类似：

  ```text
  dp[i] = max/min(dp[i-1] + f(x), dp[i-2] + g(y)) 
  ```

  例如在路径问题中：

  ```text
  dp[x][y] = min(dp[x-1][y], dp[x][y-1]) + grid[x][y]
  ```

#### 价值迭代 vs 策略迭代

- **价值迭代**类似于递归实现的动态规划，从底层子问题逐步计算状态价值。
- **策略迭代**则更像贪心+迭代的方法，在某些特定问题中更有效。

#### 应用举例

**LeetCode 经典问题的 RL 思路**

- **64. 最小路径和**：可以类比为策略迭代，从终点反向计算路径价值。
- **72. 编辑距离**：用 Bellman 方程的思想计算插入、删除、替换操作的最小成本。
- **53. 最大子序和**：可以通过状态值更新模拟一个单步决策过程。