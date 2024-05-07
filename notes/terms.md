# Artificial Intelligence











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