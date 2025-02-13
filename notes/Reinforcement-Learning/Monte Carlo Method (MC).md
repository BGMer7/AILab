Monte Carlo是[[Model Free]]的强化学习算法。这意味着它不依赖于环境的转移概率和奖励函数模型，而是通过与环境的直接交互（采样）来估计价值函数或优化策略。

Monte Carlo方法是一类通过随机采样来估计期望值的方法，在强化学习（Reinforcement Learning, RL）中被广泛用于策略评估和策略优化。与时间差分方法（Temporal Difference, TD）不同，Monte Carlo方法直接通过完整的轨迹来估计状态值或状态-动作值，无需依赖环境的动态模型。



## 应用场景

在强化学习中，我们有可能会面临以下场景：

- 环境模型未知
- 在Q-Learning或者其他方法的更新Value的公式中的状态转移函数未知，回报的概率也未知，无法直接通过计算的方式来计算一个状态的回报的函数
- 状态空间很大，导致绝大多数的情况都是可能永远不会被访问的状态
- 需要从经验中学习

例如：在图所示的正方形内部随机产生若干个点，细数落在圆中点的个数，圆的面积与正方形面积之比就等于圆中点的个数与正方形中点的个数之比。如果我们随机产生的点的个数越多，计算得到圆的面积就越接近于真实的圆的面积。
$$
\frac{圆的面积}{正方形的面积}=\frac{圆中点的面积}{正方形中点的面积}
$$
![img](https://hrl.boyuai.com/static/mc.c89f09b0.png)

蒙特卡洛方法是一种基于随机采样的数值计算方法，这里需要尤其注意的词是 **采样**

在强化学习中，主要通过真实的采样来用于估计状态价值函数和动作价值函数。其核心思想是通过多次采样和平均来逼近真实值。

蒙特卡洛方法是一种model-free的解决方案，不需要完整的环境模型（不知道模型的转移概率）就能学习。

### 应用效果

1. 策略评估（Policy Evaluation）
2. 策略改进（Policy Improvement）
3. 价值函数估计
4. 最优策略搜索



## 核心思想

**蒙特卡洛方法的核心思想是通过大量的随机样本来近似计算复杂问题的解，通过多次采样和平均来逼近真实值。**

其基本原理基于大数定律和中心极限定理，即当样本数量足够大时，样本的平均值会趋近于随机变量的期望值。

具体来说，对于一个复杂的数学问题，可以通过随机抽样来获取足够多的样本点，并通过对这些样本点的统计分析来估计问题的解。

> 大数定律是概率论中的核心概念，描述了随着试验次数的增加，样本均值会逐渐稳定并收敛到期望值。大数定律主要有两种形式：弱大数定律和强大数定律。
>
> 1. **弱大数定律**：
>
>    - 描述了随着试验次数 n 的增加，样本均值与期望值之间的差值会以概率收敛到零。
>
>    - 数学公式表示为：
>      $$
>      \lim_{n \to \infty} P\left( \left| \frac{1}{n} \sum_{i=1}^{n} X_i - \mu \right| > \epsilon \right) = 0
>      $$
>
>    - 其中，μ 是期望值，ϵ 是任意小的正数。
>
> 2. **强大数定律**：
>
>    - 描述了随着试验次数 n 的增加，样本均值会几乎必然地收敛到期望值。
>
>    - 数学公式表示为：
>      $$
>      P\left( \lim_{n \to \infty} \frac{1}{n} \sum_{i=1}^{n} X_i = \mu \right) = 1
>      $$
>
>    - 这意味着样本均值等于总体均值的概率是1。
>
> 大数定律的重要性在于它为统计推断提供了基础，确保了随着数据量的增加，我们的估计值会变得越来越可靠。
>
> 大数定律指出，随着试验次数的增加，样本均值会收敛到期望值。

### 随机抽样

随机抽样是蒙特卡洛方法的基础。随机抽样的目的是从一个已知的概率分布中生成大量的随机样本点。这些样本点可以用于模拟复杂系统的运行过程，从而获得系统的统计特性。随机抽样的方法包括：

- 均匀分布抽样：从均匀分布中生成随机样本点，例如在区间 [a, b] 内生成随机数。
- 非均匀分布抽样：从非均匀分布（如正态分布、指数分布等）中生成随机样本点，通常需要使用特定的抽样算法，如逆变换法、接受-拒绝法等。



## 核心概念

### 轨迹（Episode）

Monte Carlo方法需要完整的轨迹，即从初始状态开始到达到终止状态的一系列状态、动作和奖励组成的序列，才能进行估计。因此，它适用于具有终止状态的任务，也就是**episodic tasks**。

序列是$(s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_T)$

也有翻译 Episode 为“幕”的，一幕就是一个状态到结束时刻的T的所有的状态和动作以及回报的一个序列。

### 回报（Return)

对于时间步 $t$，其回报定义为：

$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots$

其中，$\gamma$ 为折扣因子（discount factor），用于平衡长期与短期收益。

### 估计目标

- **状态值函数 $V_\pi(s)$**：某策略 $\pi$ 下，状态 $s$ 的期望累积回报。
  $$
  V_\pi(s)=\mathbb{E}[G_t|S_t=s]
  $$

- **动作值函数 $Q_\pi(s, a)$**：状态 $s$ 下采取动作 $a$ 时的期望累积回报。
  $$
  Q_\pi(s,a)=\mathbb{E}[G_t|St=s, A_t=a]
  $$

### 首次访问型 First-Visit Monte Carlo

**定义：**
仅在轨迹中**第一次访问某个状态**时计算并更新该状态的价值。

**算法特点：**

- 跳过轨迹中对同一个状态的重复访问，只使用首次出现的状态回报。
- 避免重复统计同一轨迹中的噪声。

**更新公式：**
$$
V(s) \leftarrow V(s) + \alpha(G_t-V(s))
$$
其中：

- $G_t$ 是从首次访问该状态后的累计回报。

**优点：**

- 数据方差较小，因为只使用一次访问的信息。

**缺点：**

- 数据利用率较低，丢弃了轨迹中重复访问状态的信息。



### 每次访问型 Every-Visit Monte Carlo

**定义：**
对**轨迹中所有出现的状态**计算回报并进行更新，无论该状态是否已出现过。

**算法特点：**

- 统计轨迹中所有访问的状态，并使用所有的回报信息更新价值函数。

**更新公式：**
$$
V(s) \leftarrow V(s) + \alpha(G_t-V(s))
$$
每次访问时均计算回报 $G_t$。

**优点：**

- 数据利用率高，充分使用了轨迹中的信息。

**缺点：**

- 方差可能较大，特别是在某些噪声较多的环境中。

## 算法流程

### 首次访问和每次访问

$$
\begin{array}{l}
\textbf{First-visit MC prediction, for estimating} \ V \approx v_\pi \\

\textbf{Input:} \ \text{a policy} \ \pi \ \text{to be evaluated} \\

\textbf{Initialize:} \\
\quad V(s) \in \mathbb{R}, \text{arbitrarily, for all} \ s \in S \\
\quad \text{Returns}(s) \leftarrow \text{an empty list, for all} \ s \in S \\

\textbf{Loop forever (for each episode):} \\

\quad \text{Generate an episode following} \ \pi: \\
\quad S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_{T-1}, A_{T-1}, R_T \\

\quad G \leftarrow 0 \\

\quad \text{Loop for each step of the episode,} \ t = T - 1, T - 2, \dots, 0: \\

\quad \quad G \leftarrow \gamma G + R_{t+1} \\

\quad \quad \textbf{Unless} \ S_t \text{ appears in } S_0, S_1, \dots, S_{t-1}: \\
\quad \quad \quad \text{Append } G \text{ to } \text{Returns}(S_t) \\
\quad \quad \quad V(S_t) \leftarrow \text{average}(\text{Returns}(S_t)) \\
\end{array}
$$

#### 代码实例

```python
import numpy as np

# 模拟轨迹 [(状态, 奖励)]
episodes = [
    [(0, 1), (1, -1), (0, 2)],
    [(1, 2), (2, 3), (1, -1)],
]

gamma = 0.9
V_first_visit = np.zeros(3)
V_every_visit = np.zeros(3)
returns_first = {s: [] for s in range(3)}
returns_every = {s: [] for s in range(3)}

# 首次访问 Monte Carlo
for episode in episodes:
    G = 0
    visited = set()
    for t in reversed(range(len(episode))):
        state, reward = episode[t]
        G = reward + gamma * G
        if state not in visited:  # 首次访问条件
            visited.add(state)
            returns_first[state].append(G)
            V_first_visit[state] = np.mean(returns_first[state])

# 每次访问 Monte Carlo
for episode in episodes:
    G = 0
    for t in reversed(range(len(episode))):
        state, reward = episode[t]
        G = reward + gamma * G
        returns_every[state].append(G)
        V_every_visit[state] = np.mean(returns_every[state])

print("首次访问 MC:", V_first_visit)
print("每次访问 MC:", V_every_visit)
```



#### 平均值更新

在 Monte Carlo 方法中，**用平均值更新价值函数**的原因主要是为了获得对**期望回报**的无偏估计

我们希望估计的状态值函数 $V(s)$ 定义为：

$V(s) = \mathbb{E}_\pi [G_t | S_t = s]$

即从状态 $s$ 开始，按照策略 $\pi$ 执行时所能获得的期望累计回报 $G_t$。

由于我们无法直接计算这个数学期望，因此需要通过多次样本回报来进行估计。假设我们访问状态 $s$ $N$ 次，记录每次的回报为 $G_1, G_2, \dots, G_N$。根据大数法则：
$$
V(s) \approx \frac{1}{N} \sum_{i=1}^{N} G_i
$$
这就是**求平均值**更新的数学依据。



##### 为什么不用单次回报？

如果直接使用最后一次回报 $G_N$ 来更新 $V(s)$，可能导致：

1. **高方差问题**：单个回报可能是噪声极大的采样结果，无法准确代表状态值。
2. **收敛不稳定**：不同轨迹中的回报差异大，导致价值函数不断剧烈波动。
3. **偏差问题**：直接使用单次样本不能保证无偏估计。



##### 扩展：增量式平均公式

为了避免存储所有回报，可以用**增量更新公式**来逐步计算平均值：
$$
V(s) \leftarrow V(s) + \alpha \cdot (G - V(s))
$$
其中学习率 $\alpha = \frac{1}{N}$ 可以保证等价于求平均值：

```python
V[s] += (G - V[s]) / count
```

这种方法在大规模数据场景中更高效，且无需存储所有历史回报。





### 策略评估 (Policy Evaluation)

目标是估计给定策略 $\pi$ 的状态值或动作值函数。
**算法步骤：**

1. 初始化所有状态 $V(s)$ 或 $Q(s,a)$ 的值为任意数字（通常为零）。
2. 反复进行以下过程，直到收敛：
   - 从策略 $\pi$ 开始与环境交互，生成轨迹。
   - 对每个状态（或状态-动作对）计算回报 $G_t$。
   - 更新价值函数：
     - **首次访问（First-Visit）Monte Carlo：** 只使用每个状态的首次访问回报 $V(s) \leftarrow V(s) + \alpha (G_t - V(s))$
     - **每次访问（Every-Visit）Monte Carlo：** 使用所有访问回报

### 策略控制 (Policy Control)

目标是找到最优策略 $\pi^*$。
**方法：**

- 使用ε-greedy策略探索不同动作，从而改进策略。
- 通过贪心策略选择动作： $\pi(s) \leftarrow \arg\max_a Q(s, a)$
- 结合策略评估与改进进行策略迭代，直到收敛。

### 异策略学习 (Off-policy Learning)

Monte Carlo方法也可以在异策略学习中使用，通过**重要性采样（Importance Sampling）**校正不同策略下的样本分布：
$$
Q(s, a)\leftarrow Q(s, a)+\alpha \cdot \rho \cdot (G_t - Q(s, a))
$$
其中 $\rho$ 是重要性采样权重。



## 策略评估与优化

1. **策略评估（Policy Evaluation）**
   使用Monte Carlo方法估计给定策略 $\pi$ 的状态值或动作值函数。

2. **策略改进（Policy Improvement）**
   根据贪心策略提升（Greedy Policy Improvement），选择使得值函数最大的动作：

   $\pi(s) \leftarrow \arg\max_a Q(s, a)$

3. **策略迭代（Policy Iteration）**
   交替进行策略评估与策略改进，直到收敛。

4. **异策略学习（Off-policy Learning）**
   使用重要性采样（Importance Sampling）来校正不同策略下的采样偏差。



## Monte Carlo与其他方法的对比

| **方法**       | **依赖环境模型** | **更新方式** | **计算效率** | **优缺点**             |
| -------------- | ---------------- | ------------ | ------------ | ---------------------- |
| Monte Carlo    | 不需要           | 完整轨迹     | 较低         | 简单直观，但收敛速度慢 |
| 时间差分（TD） | 不需要           | 每步更新     | 较高         | 更高的采样效率         |
| 动态规划（DP） | 需要             | 全局更新     | 高           | 适用于已知环境模型     |



## 实际应用

1. **游戏AI**
    Monte Carlo Tree Search (MCTS) 是Monte Carlo方法的扩展，在围棋、国际象棋等策略游戏中有重要应用。
2. **机器人控制**
    Monte Carlo方法用于估计机器人路径规划中不同状态的期望回报。
3. **金融投资优化**
    在投资策略优化中用于估计长期投资收益的期望值。



需要深入理解和实现Monte Carlo强化学习方法时，可以尝试经典算法，如“Monte Carlo控制算法”和“异策略重要性采样”的代码示例。

