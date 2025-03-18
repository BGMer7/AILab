[[Advantage Function]]
[[Policy Gradient]]

## Policy Based RL的Loss Function的来源与策略梯度推导

### 一、策略梯度方法中的损失函数设计

在基于策略的强化学习（Policy-Based RL）中，损失函数并非完全人为定义，而是**通过数学推导从最大化期望回报的目标中自然导出**。其核心思想是直接优化策略参数 $\theta$ 以最大化累积奖励的期望值。以下是关键步骤：

#### 1. 目标函数定义

目标函数为期望回报：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \right]
$$

其中 $\tau = (s_0, a_0, s_1, a_1, \dots)$ 表示轨迹，$R(\tau) = \sum_{t=0}^T \gamma^t r_t$ 是折扣累积奖励。

#### 2. 策略梯度定理

通过似然比技巧（Likelihood Ratio Trick），将梯度计算转换为对策略的期望：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot Q^{\pi_\theta}(s_t,a_t) \right]
$$

其中 $Q^{\pi_\theta}(s,a)$ 是状态-动作价值函数。

#### 3. 损失函数的形式化

实际算法中，通过采样轨迹估计梯度，并定义损失函数为负的期望回报（转化为最小化问题）：

$$
L(\theta) = -\mathbb{E} \left[ \sum_{t=0}^T \log \pi_\theta(a_t|s_t) \cdot A^{\pi_\theta}(s_t,a_t) \right]
$$

此处引入**优势函数** $A(s,a) = Q(s,a) - V(s)$ 以减少方差。

---

### 二、Actor-Critic 方法中的策略梯度推导

Actor-Critic 算法通过分离策略（Actor）和价值函数（Critic）实现高效优化。其策略梯度来源于策略梯度定理，并结合Critic提供的价值估计。

#### 1. 策略梯度定理的简化

在Actor-Critic中，使用Critic估计的优势函数替代真实的 $Q$ 值：

$$
\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot A_w(s,a) \right]
$$

其中 $A_w(s,a) = Q_w(s,a) - V_w(s)$ 或更常见的单步TD误差形式：

$$
A_w(s,a) = r + \gamma V_w(s') - V_w(s)
$$

#### 2. 损失函数的具体实现

Actor的损失函数直接对应策略梯度：

$$
L_{\text{actor}}(\theta) = -\mathbb{E} \left[ \log \pi_\theta(a|s) \cdot A_w(s,a) \right]
$$

Critic的损失函数则通过最小化价值函数的误差：

$$
L_{\text{critic}}(w) = \mathbb{E} \left[ \left( r + \gamma V_w(s') - V_w(s) \right)^2 \right]
$$

---

### 三、损失函数的设计：数学推导与工程调整

#### 1. 数学推导的核心部分

- **策略梯度定理**：提供了梯度方向的理论保证，确保参数更新朝着提高期望回报的方向进行。

- **优势函数的使用**：通过引入 $A(s,a)$ 替代原始回报，显著降低方差，理论依据为：
  
  $$
  \mathbb{E}_{a \sim \pi} \left[ A(s,a) \right] = 0
  $$
  
  这使得梯度估计更稳定。

#### 2. 工程实践中的调整

- **熵正则化**：为防止策略过早收敛，添加熵项促进探索：
  
  $$
  L_{\text{actor}} = -\mathbb{E} \left[ \log \pi(a|s) \cdot A(s,a) \right] - \beta \mathbb{H}(\pi(\cdot|s))
  $$
  
  其中 $\mathbb{H}$ 是熵，$\beta$ 为系数。此项虽为启发式添加，但被广泛验证有效。

- **Clipping机制（如PPO）**：限制策略更新的幅度以避免崩溃：
  
  $$
  L^{\text{CLIP}}(\theta) = \mathbb{E} \left[ \min\left( r(\theta) A, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) A \right) \right]
  $$
  
  其中 $r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}$。这种调整虽引入人为约束，但基于对策略更新稳定性的理论分析。

---

### 四、关键公式推导示例

#### 策略梯度定理的证明概要

1. **目标函数展开**：
   
   $$
   J(\theta) = \int P(\tau|\theta) R(\tau) d\tau
   $$

2. **梯度计算**：
   
   $$
   \nabla_\theta J(\theta) = \int \nabla_\theta P(\tau|\theta) R(\tau) d\tau
   $$

3. **似然比技巧**：
   
   $$
   \nabla_\theta P(\tau|\theta) = P(\tau|\theta) \nabla_\theta \log P(\tau|\theta)
   $$

4. **轨迹概率分解**：
   
   $$
   \log P(\tau|\theta) = \sum_{t=0}^T \log \pi_\theta(a_t|s_t) + \log P(s_{t+1}|s_t,a_t)
   $$

5. **消去环境动态**：
   
   $$
   \nabla_\theta J(\theta) = \mathbb{E}_\tau \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau) \right]
   $$

6. **引入基线（Baseline）**：
   通过减去状态价值函数 $V(s)$ 得到优势函数形式，减少方差：
   
   $$
   \nabla_\theta J(\theta) = \mathbb{E}_\tau \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \left( Q(s_t,a_t) - V(s_t) \right) \right]
   $$

---

### 五、总结

- **损失函数的理论根基**：策略梯度方法中的损失函数源于最大化期望回报的数学目标，通过策略梯度定理严格推导而来，并非随意定义。
- **Actor-Critic的梯度来源**：通过分解策略与价值函数，利用Critic提供的低方差优势估计，使Actor的更新更高效。
- **工程实践中的平衡**：虽然核心部分有严格理论支持，但实际算法常引入正则化、裁剪等技巧，这些调整基于理论指导下的经验优化，以提升稳定性和效率。

```mermaid
graph LR
A[目标: 最大化期望回报 J(θ)] --> B[策略梯度定理]
B --> C[推导出梯度表达式]
C --> D[Actor损失函数: -E[logπ·A]]
B --> E[Critic价值估计]
E --> F[Critic损失函数: TD误差平方]
D --> G[工程优化: 熵正则化/Clipping]
F --> G
```
