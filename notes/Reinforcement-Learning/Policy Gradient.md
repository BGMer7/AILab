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