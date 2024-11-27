[[Policy based RL]]
[[Value based RL]]

一种特别的方法，上述两者的结合，Actor会基于概率作出动作，Critic会根据作出的动作打分，是一类结合了策略评估（Critic）和策略改进（Actor）的强化学习算法。它们通过同时学习一个策略函数（Actor）和一个值函数（Critic），从而可以更有效地学习到优秀的策略；A2C (Advantage Actor-Critic)、A3C (Asynchronous Advantage Actor-Critic)、DDPG (Deep Deterministic Policy Gradient)、TD3 (Twin Delayed Deep Deterministic Policy Gradient)、PPO (Proximal Policy Optimization)等算法均是Actor-Critic方法
————————————————

                            版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。
                        
原文链接：https://blog.csdn.net/Ever_____/article/details/133362585