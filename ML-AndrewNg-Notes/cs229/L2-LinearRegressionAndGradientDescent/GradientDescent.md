线性回归的本质就是对最小均方误差MSE求导



## Linear Model

假设线性模型为：
$$
h_\theta(x) = \theta_0 x_0 + \theta_1 x_1 + ... + \theta_j x_j
$$
模型的目标是找到参数 $\theta_0 $ 和 $ \theta_1 $ 使得预测值 $h_\theta(x) $ 和实际值 $y $ 之间的误差最小。



化简这个式子，我们假设 $x_0$ = 1，这样 $\theta$ 就是一个常系数变量。 

因此式子可以化简为：
$$
h_\theta(x) = \theta_0 + \theta_1 x_1 + ... + \theta_j x_j
$$


## Loss Function

已知MSE的公式为：
$$
MSE = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
$$
由于存在平方项，求导之后会带上一个常系数 2 ，因此在构造 loss function 的时候直接前面加上一个常系数 $\frac{1}{2}$，约分掉之后便于计算，本身损失函数乘一个常系数没有影响。

因此， 均方误差损失函数定义为：
$$
J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2
$$




>实际上在吴恩达的cs229课程中，使用的并不是最小化均方误差，而是最小化平方误差，因此在课程的公式中，实际上是没有分母m这个参数的。





## Gradient Descent

梯度下降不断更新 $\theta$ ，使得损失函数逐渐下降，更新规则如下：
$$
\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}
$$
其中： 

- $\alpha$  是学习率-learning rate，决定了每次更新的步长，这是一个可调参数
- $ \frac{\partial J(\theta)}{\partial \theta_j}$是损失函数对参数 $\theta_j $的偏导数





## Derivation Procedure

求导过程并不复杂，应用链式法则
$$
\frac{\partial J(\theta)}{\partial \theta_0} = \frac{\partial}{\partial \theta_0} \left( \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)}) \right)^2 \right) \\
  = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) \cdot \frac{\partial}{\partial \theta_i} \left( h_\theta(x^{(i)}) \right) \\
  = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) \cdot \frac{\partial}{\partial \theta_i} \left( \theta_0 + \theta_1 x_1 + ... + \theta_i x_i) \right) \\
$$
对于 $\theta_i$ 而言，由于是求偏导，因此其余的 $\theta_j$ 都不需要考虑，只需要对 $\theta_i$ 求偏导，即 $x_i$ 。

式子化简为：
$$
\frac{\partial J(\theta)}{\partial \theta_i} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}
$$
 





