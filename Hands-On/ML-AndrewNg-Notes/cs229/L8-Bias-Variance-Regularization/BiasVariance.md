# 偏差与方差

偏差和方差是统计学习理论中的两个重要概念，用于评估模型的预测误差。理解偏差和方差有助于平衡模型的复杂性和预测性能，从而提高模型的泛化能力。

## 偏差（Bias）

偏差是指模型预测值的期望与真实值之间的差异。偏差反映了模型的拟合能力，表示模型在训练数据上的表现是否准确。

- 高偏差：模型过于简单，无法捕捉数据的复杂模式，容易出现欠拟合（Underfitting）。
- 低偏差：模型能够很好地拟合训练数据，捕捉到数据的模式和特征。

偏差的数学定义为：

$$
\text{Bias} = \mathbb{E}[\hat{f}(x)] - f(x)
$$

其中，$\hat{f}(x)$ 是模型的预测函数，$f(x)$ 是数据的真实函数，$\mathbb{E}[\hat{f}(x)]$ 表示模型预测值的期望。

## 方差（Variance）

方差是指模型预测值的变化程度，反映了模型对训练数据的敏感程度。高方差的模型对训练数据中的噪声敏感，可能会在不同的训练数据集上表现出较大的差异。

- 高方差：模型过于复杂，容易对训练数据中的噪声进行过拟合（Overfitting）。
- 低方差：模型对训练数据不敏感，具有较好的泛化能力。

方差的数学定义为：

$$
\text{Variance} = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]
$$

## 偏差-方差权衡（Bias-Variance Tradeoff）

在构建机器学习模型时，偏差和方差之间存在一个权衡关系。通常，增加模型的复杂度会降低偏差，但会增加方差；相反，减少模型的复杂度会降低方差，但会增加偏差。找到一个合适的模型复杂度，以平衡偏差和方差，是构建高性能模型的关键。

总预测误差可以分解为偏差、方差和不可避免的噪声（Noise）之和：

$$
\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}
$$

## 示例

假设我们有一个简单的二次函数数据集，并使用多项式回归模型进行拟合。我们可以通过调整多项式的阶数来观察偏差和方差的变化。

1. **导入库**：

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline

    # 生成示例数据
    np.random.seed(0)
    X = np.sort(np.random.rand(100) * 10)
    y = 2 * X**2 + 3 * X + np.random.randn(100) * 10
    X = X[:, np.newaxis]
    ```

2. **定义绘图函数**：

    ```python
    def plot_polynomial_regression(X, y, degree):
        plt.scatter(X, y, color='blue', label='Data')
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, y)
        X_plot = np.linspace(0, 10, 100)[:, np.newaxis]
        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, color='red', label=f'Degree {degree}')
        plt.legend()
        plt.title(f'Polynomial Regression with Degree {degree}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.show()

    # 绘制多项式回归结果
    plot_polynomial_regression(X, y, degree=1)  # 高偏差，低方差
    plot_polynomial_regression(X, y, degree=4)  # 权衡
    plot_polynomial_regression(X, y, degree=15) # 低偏差，高方差
    ```

3. **解释结果**：

    - **Degree 1**（高偏差，低方差）：模型为线性回归，无法捕捉数据的二次关系，表现为欠拟合。
    - **Degree 4**（权衡）：模型能够较好地拟合数据，平衡了偏差和方差。
    - **Degree 15**（低偏差，高方差）：模型为高阶多项式回归，捕捉了训练数据的噪声，表现为过拟合。

## 总结

偏差和方差是评估机器学习模型性能的两个重要指标。理解并平衡偏差和方差有助于构建具有良好泛化能力的模型。在实际应用中，通常通过交叉验证和模型选择来找到最佳的偏差-方差权衡点。
