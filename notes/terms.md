# Probability and Statistics

## Terms

### Linear Regression

#### MAE

Mean Absolute Error 平均绝对误差
$$
MAE = \frac{1}{m} \sum_{i=1}^{m} |y_i - \hat{y}_i|
$$


#### MSE

Mean Squared Error 均方误差
$$
MSE = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
$$




#### RMSE

Root Mean Squared Error 均方根误差
$$
RMSE = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2}
$$
RMSE 与 MAE 的量纲相同，但求出结果后我们会发现RMSE比MAE的要大一些。

这是因为RMSE是先对误差进行**平方**的累加后再开方，它其实是放大了较大误差之间的差距。

而MAE反应的就是真实误差。因此在衡量中使RMSE的值越小其意义越大，因为它的值能反映其最大误差也是比较小的。





#### RSS

Residual Sum Of Squares

在标准线性回归中，通过最小化真实值($y_i$)和预测值($\hat{y_i}$)的平方误差。这个平方误差值也被称为残差平方和 

$RSS=\sum\limits_{i=1}^n(y_i-\hat{y_i})^2$







# Linear Algebra

#### SVD







# Machine Learning

### Loss Function



### Metrics



## Classification

### Linear Models

假设线性模型的数据集由$(x_1, y_1),(x_2, y_2), (x_3, y_3)...(x_n, y_n)$组成，其中x是自变量，y是因变量，则线性回归模型的基本形式为：

$y=w_0+w_1x_1+w_2x_2+...+w_nx_n$

vector $w$作为coef_（系数，coefficient）

$w$作为intercept_（截距）

其实还可以设置一个$\epsilon$作为误差项



#### Ordinary Least Squares

普通最小二乘法（OLS）是一种用于估计线性回归模型参数的方法。在这种方法中，我们试图找到一条直线（或者更一般地，一个超平面），使得该直线与样本数据点之间的残差平方和（即预测值与真实值之间的差的平方和）最小化。





##### Non-Negative Least Squares



#### Generalized Linear Models

减少过拟合的一个好方法是对模型进行正则化（即约束模型）：它拥有的自由度越少，则过拟合数据的难度就越大。正则化多项式模型的一种简单方法是减少多项式的次数。

对于线性模型，正则化通常是通过约束模型的权重来实现的。主要有岭回归、Lasso回归和弹性网络，它们实现了三种限制权重的方法。



##### L1 & L2 Regularization

- **L1范数**（L1 norm，曼哈顿距离或绝对值范数）： 对于一个 𝑛维向量 $X$，它的L1范数定义如下：

$$
||x||_1 = \sum_{i=1}^{n} |x_i|
$$

​		L1范数是向量中所有元素的绝对值之和。



- **L2范数**（L2 norm，欧几里得距离或平方和范数）： 对于一个 𝑛维向量 $X$，它的L2范数定义如下：

$$
||x||_2 = \sqrt{\sum_{i=1}^{n} x_i^2}
$$

​		L2范数是向量中所有元素的平方和的平方根。



#### Ridge Regression

岭回归又称为Tikhonov正则化，是线性回归的正则化版本。这迫使学习算法不仅拟合数据，而且还使模型权重尽可能小。注意仅在训练期间将正则化项添加到成本函数中。**训练完模型后，你要使用非正则化的性能度量来评估模型的性能。**
$$
min\left( \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} \beta_j^2 \right) = min \left( \frac{1}{2n} ||y - \hat{y}||_2^2 + \alpha ||\beta||_2^2 \right)
$$

其中：

- 𝑛表示样本数量。
- 𝑦表示实际观测值向量。
- 𝑦表示预测值向量。
- 𝛽表示模型的系数向量。
- 𝛼是一个调节系数，用来平衡损失函数和正则化项的重要性。**超参数α控制要对模型进行正则化的程度。如果α=0，则岭回归仅是线性回归，不具有正则化的作用。如果α非常大，则所有权重最终都非常接近于零，结果是一条经过数据均值的平线。**



Ridge采用L2范数作为penalty（惩罚项），因此也叫L2正则化。



#### Lasso Regression

线性回归的另一种正则化叫做最小绝对收缩和选择算子回归，（Least Absolute Shrinkage and Selection Operator Regression，简称Lasso回归）。
$$
min \left( \frac{1}{2n} ||y - X\beta||_2^2 + \alpha ||\beta||_1 \right)
$$
Lasso回归采用的是L1范数作为penalty（惩罚项），所以也叫L1正则化。

Lasso回归的一个重要特点是它倾向于完全消除掉最不重要特征的权重（也就是将它们设置为零）。

换句话说，Lasso回归会自动执行特征选择并输出一个稀疏模型（即只有很少的特征有非零权重）。





#### Elastic-Net





#### Bayesian Regression







#### Logistic Regression













### SVM(Support Vector Machines)



#### Kernel



#### Hyperplane

- **定义**：超平面，在一个n维空间中，一个超平面是一个n-1维的平面。例如，在二维空间中，超平面是一条直线；在三维空间中，超平面是一个平面。在高维空间中，超平面是一个(n-1)维的子空间。

- **作用**：在支持向量机中，超平面用于将特征空间划分为两个部分，每一部分对应一个类别。这样，新的数据点可以根据它们落在超平面的哪一侧来进行分类。



##### Hyperplane in SVM

- 在SVM中，我们关注的是线性可分的情况。对于二维空间中的二分类问题，超平面是一条直线。

- 对于高维空间中的数据，超平面是一个(n-1)维的子空间，可以用以下公式表示：

  $w^\mathrm Tx+b=0$

  - 其中，$w$是法向量（指向超平面的方向），$x$是数据点，$b$是偏置项。
  - 数据点$x$落在超平面的哪一侧取决于的$w^\mathrm Tx+b$符号：正数一侧为一类，负数一侧为另一类。



##### margin

在支持向量机（SVM）中，Margin（间隔）是指超平面到最近的数据点的距离。

对于线性可分的二分类问题，Margin 的计算公式可以如下表示：

对于超平面$w^\mathrm Tx+b=0$，样本$x_i$到超平面的距离可以表示为：

$$distance(x_i, Hyperplane)=\frac{|w^\mathrm Tx_i+b|}{||x||}$$

其中：

- $x_i$是数据点
- $w$是超平面的法向量
- $b$是偏置量
- $||x||$是欧几里得范数（模）



#### SVC(Support Vector Classifier)



#### LinearSVC



#### Non-LinearSVC





### KNN



### Native Bayes



### SGD Classifier









## Regression









## Clustering







## Dimensionality Reduction







## Model Selection









## Preprocessing









# Deep Learning