# Distribution

## Gaussian distribution

一元高斯分布（也称为正态分布）是一种常见的连续概率分布，在许多自然现象中有广泛的应用。

它由两个参数决定：均值（$\mu$）和方差（$\sigma^2$）。

### single dimensional Gaussian

一元高斯分布的概率密度函数（PDF）为：
$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$
其中：

- $\mu$ 是均值，表示分布的中心位置。
- $\sigma^2$ 是方差，表示分布的扩散程度。
- $\sigma$ 是标准差，是方差的平方根。

#### standard normal distribution

当高斯分布的均值 $mu = 0 $且方差 $ sigma^2 = 1$时，称为标准正态分布。标准正态分布的概率密度函数为：
$$
Z \sim N(0, 1)
$$

#### features

1. **对称性**：高斯分布关于均值 $\mu$ 对称。
2. **峰值**：在均值 $\mu$ 处达到最大值。
3. **68-95-99.7 规则**：在均值的 $\pm 1\sigma$、$\pm 2\sigma$ 和 $\pm 3\sigma$ 范围内，包含了大约 68%、95% 和 99.7% 的数据。



### multivariate Gaussian distribution

多元高斯分布是单变量高斯分布在高维空间中的推广。它用于描述一个向量的联合分布，其中每个元素都服从正态分布，并且这些元素之间可能存在相关性。

#### 

对于随机向量 $\mathbf{X} = [X_1, X_2, \ldots, X_n]^T$，其服从均值向量 $\boldsymbol{\mu}$ 和协方差矩阵 $\boldsymbol{\Sigma}$ 的多元高斯分布，其概率密度函数 (PDF) 表示为：

$$
f(\mathbf{x}) = \frac{1}{(2\pi)^{n/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
$$

其中：
- $\mathbf{x} $ 是一个 $ n $ 维向量 
- $\boldsymbol{\mu} $ 是 $ \mathbf{X} $ 的均值向量
- $\boldsymbol{\Sigma} $ 是 $ \mathbf{X} $ 的 $ n \times n $ 协方差矩阵
- $|\boldsymbol{\Sigma}| $ 是协方差矩阵的行列式
- $\boldsymbol{\Sigma}^{-1} $ 是协方差矩阵的逆

#### features

1. **线性变换**：如果 $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，并且 $\mathbf{A}$ 是一个矩阵，$\mathbf{b}$ 是一个向量，则 $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$ 也服从高斯分布，即 $\mathbf{Y} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T)$。
2. **边缘分布**：若 $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，则其任意子集的边缘分布仍然是高斯分布。
3. **条件分布**：若 $\mathbf{X}$ 可以分解为 $\mathbf{X} = [\mathbf{X}_1, \mathbf{X}_2]^T$，则在给定 $\mathbf{X}_2 = \mathbf{x}_2$ 的条件下，$\mathbf{X}_1$ 的条件分布也是高斯分布。



#### examples

假设我们有一个二维随机向量 $\mathbf{X} = [X_1, X_2]^T$，其均值向量 $\boldsymbol{\mu}$ 和协方差矩阵 $\boldsymbol{\Sigma}$ 分别为：

$$
\boldsymbol{\mu} = \begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix}, \quad
\boldsymbol{\Sigma} = \begin{bmatrix} \sigma_1^2 & \rho \sigma_1 \sigma_2 \\ \rho \sigma_1 \sigma_2 & \sigma_2^2 \end{bmatrix}
$$

则其概率密度函数为：

$$
f(\mathbf{x}) = \frac{1}{2\pi |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
$$

