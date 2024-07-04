# 支持向量机（SVM）原理

支持向量机（Support Vector Machine, SVM）是一种用于分类和回归的监督学习模型。SVM的核心思想是通过找到一个最佳超平面来分隔不同类别的数据点，以最大化类别之间的间隔（Margin）。

## 主要概念

### 超平面

在二维空间中，超平面是一条线，而在高维空间中，超平面是一个维度比空间维度低一维的平面。例如，在三维空间中，超平面是一个二维平面。

### 间隔

间隔是指从数据点到超平面的最短距离。SVM通过最大化这个间隔来找到最优的超平面，使得模型对噪声和误差具有更好的鲁棒性。

### 支持向量

支持向量是指距离超平面最近的数据点。这些点决定了超平面的位置和方向。



### 支持向量机中的函数间隔和几何间隔

在支持向量机（SVM）中，函数间隔（Functional Margin）和几何间隔（Geometric Margin）是两个关键概念，它们描述了数据点与分类超平面之间的距离，并用于确定最优超平面的参数。

#### 函数间隔（Functional Margin）

函数间隔是指数据点在超平面上的投影与分类标签的乘积。对于给定的训练样本 $\{(\mathbf{x}_i, y_i)\}$，超平面参数为 $\mathbf{w}$ 和 $b$，函数间隔定义为：

$$
\hat{\gamma}_i = y_i (\mathbf{w} \cdot \mathbf{x}_i + b)
$$

其中，$y_i$ 是样本 $\mathbf{x}_i$ 的标签，$\mathbf{w}$ 是超平面的法向量，$b$ 是偏置项。函数间隔衡量的是样本点在超平面一侧的距离和分类正确性。

为了保证所有样本点都正确分类，要求函数间隔 $\hat{\gamma}_i > 0$。

#### 几何间隔（Geometric Margin）

几何间隔是指数据点到超平面的最短距离。几何间隔定义为：

$$
\gamma_i = \frac{\hat{\gamma}_i}{||\mathbf{w}||} = \frac{y_i (\mathbf{w} \cdot \mathbf{x}_i + b)}{||\mathbf{w}||}
$$

几何间隔是函数间隔标准化后的结果，消除了法向量 $\mathbf{w}$ 的影响。几何间隔表示数据点到超平面的实际欧几里得距离。

为了最大化几何间隔，我们需要最大化以下优化目标：

$$
\max_{\mathbf{w}, b} \min_i \gamma_i = \frac{1}{||\mathbf{w}||} \max_{\mathbf{w}, b} \min_i y_i (\mathbf{w} \cdot \mathbf{x}_i + b)
$$

##### 关系和优化目标

函数间隔和几何间隔之间的关系为：

$$
\gamma_i = \frac{\hat{\gamma}_i}{||\mathbf{w}||}
$$

几何间隔最大化相当于在超平面参数 $\mathbf{w}$ 和 $b$ 的空间中寻找一个超平面，使得最小几何间隔最大化。SVM的目标是找到一个超平面，使得几何间隔最大化，从而提高分类器的鲁棒性。

在SVM的优化过程中，通常通过以下约束来规范化几何间隔：

$$
y_i (\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1
$$

在此约束下，几何间隔可以表示为：

$$
\gamma_i = \frac{1}{||\mathbf{w}||}
$$

##### 示例

假设我们有一个二分类问题，数据集为 $\mathbf{X} = \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N\}$，类别标签为 $\mathbf{y} = \{y_1, y_2, \ldots, y_N\}$。我们使用SVM进行分类，并计算函数间隔和几何间隔。

1. **导入库**：

    ```python
    from sklearn.svm import SVC
    import numpy as np

    # 生成一些示例数据
    np.random.seed(0)
    X = np.random.randn(100, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int) * 2 - 1  # 转换为 {-1, 1}
    ```

2. **训练SVM模型**：

    ```python
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X, y)
    ```

3. **计算函数间隔和几何间隔**：

    ```python
    w = svm.coef_[0]
    b = svm.intercept_[0]
    
    functional_margins = y * (X.dot(w) + b)
    geometric_margins = functional_margins / np.linalg.norm(w)
    
    print("Functional Margins: ", functional_margins)
    print("Geometric Margins: ", geometric_margins)
    ```



## SVM的优化目标

SVM的目标是找到一个超平面，使得分类间隔最大化。对于线性可分的数据集，优化目标可以表示为：

$$
\max_{\mathbf{w}, b} \frac{2}{||\mathbf{w}||}
$$

其中，$\mathbf{w}$ 是超平面的法向量，$b$ 是偏置项。

在实际应用中，我们常常使用对偶问题来解决这个优化问题，通过引入拉格朗日乘子 $\alpha_i$，优化目标变为：

$$
\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i \cdot \mathbf{x}_j
$$

约束条件为：

$$
\sum_{i=1}^n \alpha_i y_i = 0 \quad \text{和} \quad \alpha_i \geq 0 \quad \forall i
$$

## 核函数

对于非线性可分的数据，SVM通过使用核函数将数据映射到高维空间，在高维空间中找到一个线性超平面。常见的核函数包括线性核、多项式核、径向基函数（RBF）核和Sigmoid核。

### 核技巧（Kernel Trick）

核技巧的基本思想是利用核函数计算在高维空间中进行点积运算，而无需显式地进行高维映射。具体来说，对于一个非线性映射 $\phi: \mathbb{R}^n \rightarrow \mathbb{R}^m$，核函数 $K(\mathbf{x}_i, \mathbf{x}_j)$ 定义为：

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j)
$$

常见的核函数包括：

1. **线性核**：$K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i \cdot \mathbf{x}_j$
2. **多项式核**：$K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i \cdot \mathbf{x}_j + c)^d$
3. **高斯径向基函数（RBF）核**：$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left( -\frac{||\mathbf{x}_i - \mathbf{x}_j||^2}{2\sigma^2} \right)$
4. **Sigmoid核**：$K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\alpha \mathbf{x}_i \cdot \mathbf{x}_j + c)$

## SVM分类器

SVM分类器的决策函数为：

$$
f(\mathbf{x}) = \text{sign}(\sum_{i=1}^n \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b)
$$

其中，$\alpha_i$ 是拉格朗日乘子，$y_i$ 是类别标签，$K(\mathbf{x}_i, \mathbf{x})$ 是核函数。

## SVM回归（SVR）

支持向量回归（Support Vector Regression, SVR）是SVM的一种扩展，用于回归问题。SVR的目标是找到一个函数，使得大部分数据点的预测值与真实值的差异在 $\epsilon$ 范围内，并且尽量平滑函数。

SVR的优化目标为：

$$
\min_{\mathbf{w}, b} \frac{1}{2} ||\mathbf{w}||^2 + C \sum_{i=1}^n (\xi_i + \xi_i^*)
$$

约束条件为：

$$
\begin{aligned}
y_i - (\mathbf{w} \cdot \mathbf{x}_i + b) &\leq \epsilon + \xi_i \\
(\mathbf{w} \cdot \mathbf{x}_i + b) - y_i &\leq \epsilon + \xi_i^* \\
\xi_i, \xi_i^* &\geq 0
\end{aligned}
$$

其中，$\xi_i$ 和 $\xi_i^*$ 是松弛变量，$C$ 是正则化参数。

## 示例代码

下面是一个使用SVM进行分类的示例代码，使用Python和scikit-learn库：

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

# 生成一些示例数据
np.random.seed(0)
X = np.random.randn(100, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义参数网格
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
}

# 初始化SVM模型和网格搜索
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 输出最优参数和模型性能
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy: ", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Test set accuracy: ", accuracy_score(y_test, y_pred))