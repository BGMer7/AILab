[[Value Function]]

学习最优动作价值函数Q，无需策略函数，如Q-learning，Sarsa，DQN方法

原文链接：https://blog.csdn.net/Ever_____/article/details/133362585

**Value-based 强化学习**是强化学习中的一种方法，主要通过估计某一策略下的状态价值函数或动作价值函数来求解最优策略。它的核心思想是通过对环境的价值进行建模，以此来引导智能体选择最优的动作。

在 **Value-based** 强化学习中，智能体通过学习一个与状态（或状态-动作对）相关的函数来评估各状态的“好坏”，而无需直接建模或优化策略。

## 价值函数[[Value Function]]