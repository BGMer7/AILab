[[Policy based RL]]
策略梯度（Policy Gradient）方法是强化学习中直接对策略进行优化的一类算法。

基于值函数的方法不同，策略梯度方法通过参数化策略，并通过优化这些参数来最大化预期回报。 

## 策略表示

假设策略由参数 $\theta$ 控制，表示为 $\pi_\theta(a|s)$，表示在状态 $s$ 下采取动作 $a$ 的概率。我们的目标是找到最优参数 $\theta$，使得从初始状态开始的预期总回报 $J(\theta)$ 最大化。 

## 目标函数

目标是最大化策略的预期回报，定义为：

$$
J(θ)=\mathbb{E}_{τ∼πθ}[R(τ)]
$$

其中，$\tau$ 表示一个轨迹（即状态-动作序列），$R(\tau)$ 是轨迹的总回报。

## 策略梯度定理

为了优化 $J(\theta)$，需要计算其梯度 $\nabla_\theta J(\theta)$。根据策略梯度定理，该梯度可以表示为：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) R(\tau) \right]
$$

该公式表明，可以通过对数概率的梯度与总回报的乘积的期望来估计策略梯度。 

## REINFORCE算法

REINFORCE是最基本的策略梯度算法，其更新规则为：

$$
\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) R(\tau)
$$

中，$\alpha$ 是学习率。算法通过采样轨迹，计算每个时间步的梯度，并根据总回报对策略参数进行更新。 

## 引入基准线

为了减少梯度估计的方差，可以引入一个基准线（baseline）$b(s)$，更新规则变为：

$$
\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \left( R(\tau) - b(s_t) \right)
$$

常见的选择是使用状态价值函数 $V(s)$ 作为基准线，此时 $R(\tau) - V(s_t)$ 即为优势函数 $A(s_t, a_t)$。 

策略梯度方法通过直接对策略进行参数化，并使用梯度上升法来优化策略参数，以最大化预期回报。引入基准线可以有效降低梯度估计的方差，提高算法的稳定性和收敛速度。



# 策略梯度方法数学公式

## 基本概念

策略梯度是强化学习中一类重要的算法，它直接对策略进行参数化并优化。不同于基于值函数的方法，策略梯度方法直接学习从状态到动作的映射。

## 核心数学公式

### 目标函数

策略梯度方法的目标是最大化期望回报：
$$
J(\theta) = \mathbb{E}*{\tau \sim p*{\theta}(\tau)} \left[ \sum_{t=0}^{T} r(s_t, a_t) \right]
$$
其中：

- $\theta$ 是策略参数
- $\tau$ 是轨迹 $(s_0, a_0, s_1, a_1, ..., s_T, a_T)$
- $p_{\theta}(\tau)$ 是在策略 $\pi_{\theta}$ 下生成轨迹 $\tau$ 的概率
- $r(s_t, a_t)$ 是在状态 $s_t$ 执行动作 $a_t$ 获得的即时奖励

### 策略梯度定理

策略梯度定理给出了目标函数梯度的计算方法：
$$
\nabla_{\theta} J(\theta) = \mathbb{E}*{\tau \sim p*{\theta}(\tau)} \left[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \cdot R_t \right]
$$
其中：

- $\nabla_{\theta} \log \pi_{\theta}(a_t|s_t)$ 是策略对数概率的梯度
- $R_t = \sum_{k=t}^{T} r(s_k, a_k)$ 是从时间步 $t$ 开始的累积回报

### REINFORCE 算法

REINFORCE 算法是策略梯度方法的一种实现：
$$
\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \cdot R_t
$$
其中 $\alpha$ 是学习率。

### 基线减法

为了减少方差，通常引入基线函数：
$$
\nabla_{\theta} J(\theta) = \mathbb{E}*{\tau \sim p*{\theta}(\tau)} \left[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \cdot (R_t - b(s_t)) \right]
$$
其中 $b(s_t)$ 是基线函数，通常设为值函数 $V(s_t)$。

### 优势函数

进一步引入优势函数可以更有效地降低方差：
$$
 A(s_t, a_t) = Q(s_t, a_t) - V(s_t) 
$$
其中：

- $Q(s_t, a_t)$ 是状态-动作值函数
- $V(s_t)$ 是状态值函数

### Actor-Critic 方法

Actor-Critic 方法同时学习策略和值函数：
$$
\nabla_{\theta} J(\theta) = \mathbb{E}*{\tau \sim p*{\theta}(\tau)} \left[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \cdot A(s_t, a_t) \right]
$$

### 自然策略梯度

自然策略梯度使用Fisher信息矩阵来调整更新方向：
$$
\theta \leftarrow \theta + \alpha F^{-1} \nabla_{\theta} J(\theta)
$$
其中 $F$ 是Fisher信息矩阵：
$$
F = \mathbb{E}*{\tau \sim p*{\theta}(\tau)} \left[ \nabla_{\theta} \log \pi_{\theta}(\tau) \nabla_{\theta} \log \pi_{\theta}(\tau)^T \right]
$$

### 近端策略优化 (PPO)

PPO 算法使用裁剪目标函数来限制策略更新幅度：
$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right]
$$
其中：

- $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是新旧策略的概率比
- $\epsilon$ 是裁剪参数（通常设为0.1或0.2）

## 实际应用

策略梯度方法在连续控制、游戏AI和机器人学习等领域有广泛应用。通过直接优化策略参数，这类方法能够处理高维连续动作空间，为复杂决策问题提供有效解决方案。
