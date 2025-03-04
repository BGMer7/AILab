## **1. 介绍**

在深度学习模型中，归一化（Normalization）是稳定训练、加速收敛和防止梯度消失/爆炸的重要方法。以下是三种常见的归一化方法：

| 归一化方法                      | 计算范围                                                     | 适用场景                              |
| ------------------------------- | ------------------------------------------------------------ | ------------------------------------- |
| **Batch Normalization (BN)**    | 以 **Batch** 维度归一化（对 batch 维度计算均值和方差）       | 适用于 CNN、MLP，大 batch 训练        |
| **Layer Normalization (LN)**    | 以 **单个样本的特征维度** 归一化（每个样本独立计算均值和方差） | 适用于 NLP、RNN、Transformer          |
| **RMS Normalization (RMSNorm)** | 计算 **特征维度的均方根 (RMS)** 进行归一化                   | 适用于 NLP、Transformer，减少均值计算 |

------

## **2. Batch Normalization (BN)**

### **2.1 BN 原理**

Batch Normalization（批归一化）通过在**批次维度**（mini-batch）计算均值和标准差，对输入进行标准化：
$$
\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i
$$

$$
\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2
$$


$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

$$
y_i = \gamma \hat{x}_i + \beta
$$


其中：

- $\mu_B$, $\sigma_B^2$ 是当前 batch 内的均值和方差。
- $\gamma$, $\beta$ 是**可学习参数**（Scale 和 Shift）。
- $\epsilon$ 是一个小数，防止分母为零。

### **2.2 BN 代码示例**

```python
import torch
import torch.nn as nn

# 3通道的 2D 批归一化
bn = nn.BatchNorm2d(num_features=3)

# 生成一个 4x3x32x32 的输入
x = torch.rand(4, 3, 32, 32)  # batch_size=4, channels=3, height=32, width=32
output = bn(x)

print(output.shape)  # (4, 3, 32, 32)
```

### **2.3 BN 的优缺点**

✅ **优点**

- 适用于 CNN 和 MLP，可以加速收敛。
- 通过规范化，使得不同 batch 之间的输入分布更加稳定。

❌ **缺点**

- **依赖 batch 统计量**，小 batch 训练效果较差。
- 训练和推理模式不同（训练时用 batch 均值，推理时用全局均值）。

------

## **3. Layer Normalization (LN)**

### **3.1 LN 原理**

Layer Normalization（层归一化）在 **单个样本的所有特征维度** 进行归一化（而不是在 batch 维度），它适用于 NLP、RNN、Transformer。

计算方式：

$$
\mu_L = \frac{1}{n} \sum_{j=1}^{n} x_j
$$

$$
\sigma_L^2 = \frac{1}{n} \sum_{j=1}^{n} (x_j - \mu_L)^2
$$

$$
\hat{x}_j = \frac{x_j - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}}
$$

$$
y_j = \gamma \hat{x}_j + \beta
$$

其中：

- $\mu_L$, $\sigma_L^2$ 是**当前样本**的均值和方差。
- $\gamma$, $\beta$ 是可学习参数。

### **3.2 LN 代码示例**

```python
ln = nn.LayerNorm(normalized_shape=10)  # 归一化 feature 维度

x = torch.rand(4, 10)  # batch_size=4, feature_dim=10
output = ln(x)

print(output.shape)  # (4, 10)
```

### **3.3 LN 的优缺点**

✅ **优点**

- 不依赖 batch 统计量，可用于小 batch 训练。
- 适用于 **变长序列（如 NLP、Transformer）**。

❌ **缺点**

- 在 CNN 中效果不如 BatchNorm，因为不同通道的特征可能不应该归一化到相同的分布。

------

## **4. RMS Normalization (RMSNorm)**

### **4.1 RMSNorm 原理**

RMS Normalization 是 LayerNorm 的变体，它使用 **均方根（RMS）** 进行归一化，而不是计算均值和标准差。

公式：

$$
RMS(x) = \sqrt{\frac{1}{n} \sum_{j=1}^{n} x_j^2}
$$

$$
\hat{x}_j = \frac{x_j}{RMS(x) + \epsilon}
$$

$$
y_j = \gamma \hat{x}_j
$$

**与 LayerNorm 的区别：**

- **不计算均值**，仅考虑均方根 RMS。
- 计算量更少（无均值减法操作），适合 NLP 任务。

### **4.2 RMSNorm 代码示例**

PyTorch **没有原生 RMSNorm**，可以用 `torch.norm()` 自定义：

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.norm(x, dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        return self.gamma * (x / (rms + self.eps))

# 10 维特征的 RMSNorm
rms_norm = RMSNorm(10)

x = torch.rand(4, 10)  # batch_size=4, feature_dim=10
output = rms_norm(x)

print(output.shape)  # (4, 10)
```

### **4.3 RMSNorm 的优缺点**

✅ **优点**

- 计算量比 LayerNorm **更少**。
- 适用于 Transformer，特别是 NLP 模型。

❌ **缺点**

- 适用于 NLP，不适用于 CNN。
- 归一化方式不考虑均值，可能影响某些任务的稳定性。

------

## **5. 总结对比**

| 归一化方法    | 归一化范围         | 适用场景              | 是否依赖 batch |
| ------------- | ------------------ | --------------------- | -------------- |
| **BatchNorm** | batch 维度         | CNN、MLP              | ✅ 依赖 batch   |
| **LayerNorm** | 每个样本的特征维度 | RNN、Transformer、NLP | ❌ 不依赖 batch |
| **RMSNorm**   | 仅用均方根         | NLP、Transformer      | ❌ 不依赖 batch |

------

## **6. 什么时候用哪种归一化？**

- **CNN**: **BatchNorm**（更快、更稳定）
- **NLP / Transformer**: **LayerNorm** 或 **RMSNorm**
- **小 batch 训练**: **LayerNorm / RMSNorm**（BatchNorm 依赖 batch 统计量）
- **计算量优化**: **RMSNorm**（不计算均值）

🚀 **动手练习建议**

1. **对 Transformer 模型替换 `LayerNorm` 和 `RMSNorm`，对比性能和梯度稳定性。**
2. **使用小 batch 训练 CNN，尝试 `BatchNorm` 和 `LayerNorm`，观察训练速度变化。**