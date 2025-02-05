Policy Iteration的目的是通过迭代计算value function 价值函数的方式来使policy收敛到最优。

Policy Iteration本质上就是直接使用Bellman方程而得到的：
$$
\begin{aligned}
v_{k+1}(s) &\doteq \mathbb{E}_{\pi} \left[ R_{t+1} + \gamma v_k(S_{t+1}) \mid S_t = s \right] \\
&= \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_k(s') \right]
\end{aligned}
$$
那么Policy Iteration一般分成两步：

1. Policy Evaluation 策略评估。目的是 更新Value Function，或者说更好的估计基于当前策略的价值
2. Policy Improvement 策略改进。 使用 greedy policy 产生新的样本用于第一步的策略评估。

![img](https://pic3.zhimg.com/aeb96f036bea2860a8e1cbde2eef4c04_r.jpg)

本质上就是使用当前策略产生新的样本，然后使用新的样本更好的估计策略的价值，然后利用策略的价值更新策略，然后不断反复。理论可以证明最终策略将收敛到最优。
具体算法：

$$
\begin{aligned}
&\textbf{1. Initialization} \\
&V(s) \in \mathbb{R} \quad \text{and} \quad \pi(s) \in A(s) \quad \text{arbitrarily for all} \quad s \in S \\

&\textbf{2. Policy Evaluation} \\
&\text{Repeat:} \\
&\quad \Delta \leftarrow 0 \\
&\quad \text{For each} \ s \in S: \\
&\qquad v \leftarrow V(s) \\
&\qquad V(s) \leftarrow \sum_{s', r} p(s', r | s, \pi(s)) \left[ r + \gamma V(s') \right] \\
&\qquad \Delta \leftarrow \max(\Delta, | v - V(s) |) \\
&\text{until} \ \Delta < \theta \ (\text{a small positive number}) \\

&\textbf{3. Policy Improvement} \\
&\text{policy-stable} \leftarrow \text{true} \\
&\text{For each} \ s \in S: \\
&\quad a \leftarrow \pi(s) \\
&\quad \pi(s) \leftarrow \arg\max_a \sum_{s', r} p(s', r | s, a) \left[ r + \gamma V(s') \right] \\
&\quad \text{If} \ a \neq \pi(s): \\
&\qquad \text{policy-stable} \leftarrow \text{false} \\
&\text{If policy-stable, stop and return} \ V \ \text{and} \ \pi \\
&\text{else go to 2}
\end{aligned}
$$
那么这里要注意的是policy evaluation部分。这里的迭代很重要的一点是需要知道state[状态转移概率](https://zhida.zhihu.com/search?content_id=735216&content_type=Article&match_order=1&q=状态转移概率&zhida_source=entity)p。也就是说依赖于model模型。而且按照算法要反复迭代直到收敛为止。所以一般需要做限制。比如到某一个比率或者次数就停止迭代。那么需要特别说明的是不管是策略迭代还是值迭代都是在理想化的情况下（上帝视角）推导出来的算法，本质上并不能直接应用，因为依赖Model。