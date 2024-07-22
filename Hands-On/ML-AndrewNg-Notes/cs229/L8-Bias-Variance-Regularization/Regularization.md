# 线性回归的正则化

正则化是防止模型过拟合的重要技术。它通过在损失函数中添加惩罚项来限制模型的复杂度，从而提高模型的泛化能力。在线性回归中，常见的正则化方法有岭回归（L2正则化）和Lasso回归（L1正则化）。

## 岭回归（Ridge Regression）

岭回归在普通最小二乘法的损失函数中添加了一个L2正则化项。L2正则化通过惩罚权重的平方和来限制权重的大小，从而防止过拟合。

- **损失函数**：

$$ 
J(\mathbf{w}) = \sum_{i=1}^{m} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \lambda \sum_{j=1}^{n} w_j^2 
$$

其中，$\lambda$ 是正则化参数，用于控制惩罚项的权重。较大的 $\lambda$ 值会强制权重更接近于零，从而减少模型的复杂度。

- **优化目标**：

$$ 
\min_{\mathbf{w}} \left( \sum_{i=1}^{m} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \lambda \sum_{j=1}^{n} w_j^2 \right) 
$$

## Lasso回归（Lasso Regression）

Lasso回归在普通最小二乘法的损失函数中添加了一个L1正则化项。L1正则化通过惩罚权重的绝对值和来限制权重的大小，从而防止过拟合。Lasso回归还有一个特点，它可以产生稀疏模型，即一些权重会被强制为零，从而实现特征选择。

- **损失函数**：

$$ 
J(\mathbf{w}) = \sum_{i=1}^{m} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \lambda \sum_{j=1}^{n} |w_j| 
$$

其中，$\lambda$ 是正则化参数，用于控制惩罚项的权重。较大的 $\lambda$ 值会强制更多的权重为零，从而实现特征选择。

- **优化目标**：

$$ 
\min_{\mathbf{w}} \left( \sum_{i=1}^{m} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \lambda \sum_{j=1}^{n} |w_j| \right) 
$$

## 弹性网络（Elastic Net）

弹性网络结合了L2正则化和L1正则化，通过在损失函数中同时添加L2和L1正则化项来限制权重的大小。弹性网络在处理高维数据和具有多重共线性的特征时表现良好。

- **损失函数**：

$$ 
J(\mathbf{w}) = \sum_{i=1}^{m} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \lambda_1 \sum_{j=1}^{n} |w_j| + \lambda_2 \sum_{j=1}^{n} w_j^2 
$$

其中，$\lambda_1$ 和 $\lambda_2$ 是正则化参数，用于控制L1和L2正则化项的权重。

- **优化目标**：

$$ 
\min_{\mathbf{w}} \left( \sum_{i=1}^{m} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \lambda_1 \sum_{j=1}^{n} |w_j| + \lambda_2 \sum_{j=1}^{n} w_j^2 \right) 
$$

## 正则化中的 $\lambda$ 的意义

在机器学习中的正则化技术中，$\lambda$ 是一个超参数，用于控制正则化项在损失函数中的权重。正则化项的作用是限制模型的复杂度，防止过拟合。通过调整 $\lambda$ 的值，可以在模型的复杂度和泛化能力之间找到最佳平衡点。

### 控制模型复杂度

- **较大的 $\lambda$**：当 $\lambda$ 值较大时，正则化项的权重较大，模型会更倾向于缩小权重参数。这会使模型变得更加简单，从而减少过拟合风险。但如果 $\lambda$ 过大，可能会导致欠拟合，因为模型过于简单，无法捕捉数据的复杂模式。
- **较小的 $\lambda$**：当 $\lambda$ 值较小时，正则化项的权重较小，模型会更倾向于拟合训练数据，可能导致过拟合，因为模型在训练数据上表现很好，但在测试数据上表现较差。

### 偏差-方差权衡

- **高 $\lambda$**：增加偏差，减少方差。模型变得更简单，偏差增大，但方差减小，泛化能力提高。
- **低 $\lambda$**：减少偏差，增加方差。模型变得更复杂，偏差减小，但方差增大，可能导致过拟合。

通过选择合适的 $\lambda$ 值，可以在偏差和方差之间找到最佳的平衡点，从而优化模型的泛化性能。

### 选择 $\lambda$ 的方法

在实践中，$\lambda$ 通常通过交叉验证来选择。交叉验证是一种评估模型性能的方法，通过在不同的数据子集上训练和测试模型，以找到最佳的 $\lambda$ 值。

## 示例代码

以下是使用scikit-learn库进行交叉验证以选择最佳 $\lambda$ 值的示例：

```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, train_test_split

# 生成示例数据
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 岭回归的超参数选择
ridge = Ridge()
param_grid = {'alpha': [0.1, 1, 10, 100]}
ridge_cv = GridSearchCV(ridge, param_grid, cv=5)
ridge_cv.fit(X_train, y_train)
print("Best lambda for Ridge Regression:", ridge_cv.best_params_['alpha'])

# Lasso回归的超参数选择
lasso = Lasso()
param_grid = {'alpha': [0.01, 0.1, 1, 10]}
lasso_cv = GridSearchCV(lasso, param_grid, cv=5)
lasso_cv.fit(X_train, y_train)
print("Best lambda for Lasso Regression:", lasso_cv.best_params_['alpha'])

# 弹性网络回归的超参数选择
elastic_net = ElasticNet()
param_grid = {'alpha': [0.01, 0.1, 1, 10], 'l1_ratio': [0.2, 0.5, 0.8]}
elastic_net_cv = GridSearchCV(elastic_net, param_grid, cv=5)
elastic_net_cv.fit(X_train, y_train)
print("Best parameters for Elastic Net:", elastic_net_cv.best_params_)
