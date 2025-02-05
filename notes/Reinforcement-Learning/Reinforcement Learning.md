[[Action]]
[[Agent]]
[[Reward]]
[[State]]
[[Policy]]
[[Supervised Learning]]
[[Markov Decision Process(MDP)]]
[[Markov Reward Process(MRP)]]
[[Actor-Critic]]
[[Policy based RL]]
[[Value based RL]]



# RL和SL的区别
此外，RL和监督学习（supervised learning）的区别：

监督学习有标签告诉算法什么样的输入对应着什么样的输出（譬如分类、回归等问题）
所以对于监督学习，目标是找到一个最优的模型函数，使其在训练数据集上最小化一个给定的损失函数，相当于最小化预测误差
最优模型 = arg minE { [损失函数(标签,模型(特征)] }

RL没有标签告诉它在某种情况下应该做出什么样的行为，只有一个做出一系列行为后最终反馈回来的reward，然后判断当前选择的行为是好是坏
相当于RL的目标是最大化智能体策略在和动态环境交互过程中的价值，而策略的价值可以等价转换成奖励函数的期望，即最大化累计下来的奖励期望
最优策略 = arg maxE { [奖励函数(状态,动作)] }
 
监督学习如果做了比较坏的选择则会立刻反馈给算法
RL的结果反馈有延时，有时候可能需要走了很多步以后才知道之前某步的选择是好还是坏
监督学习中输入是独立分布的，即各项数据之间没有关联
RL面对的输入总是在变化，每当算法做出一个行为，它就影响了下一次决策的输入

## Refs
[DQN 从入门到放弃1 DQN与增强学习 - 知乎](https://zhuanlan.zhihu.com/p/21262246?refer=intelligentunit)
[DQN 从入门到放弃2 增强学习与MDP - 知乎](https://zhuanlan.zhihu.com/p/21292697?refer=intelligentunit)
[DQN 从入门到放弃3 价值函数与Bellman方程 - 知乎](https://zhuanlan.zhihu.com/p/21340755)
[DQN 从入门到放弃4 动态规划与Q-Learning - 知乎](https://zhuanlan.zhihu.com/p/21378532)
[DQN从入门到放弃5 深度解读DQN算法 - 知乎](https://zhuanlan.zhihu.com/p/21421729)