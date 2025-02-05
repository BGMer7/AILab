[[Q-Learning]]

[强化学习(七)--Q-Learning和Sarsa - 知乎](https://zhuanlan.zhihu.com/p/46850008)

## Q-Learning 和 SARSA

Q-Learning的目的是学习特定State下、特定Action的价值。是建立一个Q-Table，以State为行、Action为列，通过每个动作带来的奖赏更新Q-Table。

Q-Learning是[off-policy](https://zhida.zhihu.com/search?content_id=9473281&content_type=Article&match_order=1&q=off-policy&zhida_source=entity)的。异策略是指行动策略和评估策略不是一个策略。Q-Learning中行动策略是ε-greedy策略，要更新Q表的策略是贪婪策略。

Sarsa全称是state-action-reward-state'-action'。 也是采用Q-table的方式存储[动作值函数](https://zhida.zhihu.com/search?content_id=9473281&content_type=Article&match_order=1&q=动作值函数&zhida_source=entity)；而且决策部分和Q-Learning是一样的, 也是采用ε-greedy策略。不同的地方在于 Sarsa 的更新方式是不一样的。

1. Sarsa是[on-policy](https://zhida.zhihu.com/search?content_id=9473281&content_type=Article&match_order=1&q=on-policy&zhida_source=entity)的更新方式，它的行动策略和评估策略都是ε-greedy策略。
2. Sarsa是先做出动作后更新。

Q-Learning算法，先假设下一步选取最大奖赏的动作，更新值函数。然后再通过ε-greedy策略选择动作。

Sarsa算法，先通过ε-greedy策略执行动作，然后根据所执行的动作，更新值函数。