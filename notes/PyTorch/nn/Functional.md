# PyTorch `torch.nn.functional` 详解

## 1. `torch.nn.functional` 简介

`torch.nn.functional`（简称 `F`）是 PyTorch 提供的一组 **无状态**（stateless）的函数接口，用于执行各种神经网络操作，如激活函数、损失函数、归一化、卷积、池化等。这些函数通常用于 `torch.nn.Module` 中的 `forward()` 方法，而不是作为 `nn.Module` 的一部分。

它提供了许多函数式操作，用于构建和训练神经网络。与 `torch.nn` 中的类不同，`torch.nn.functional` 中的函数是无状态的，不包含可学习的参数，而是直接对输入张量进行操作。

### `torch.nn.Module` vs. `torch.nn.functional`

- `torch.nn.Module` 方式（带状态）：

  ```python
  import torch.nn as nn
  relu = nn.ReLU()
  output = relu(input)
  ```

- `torch.nn.functional` 方式（无状态）：

  ```python
  import torch.nn.functional as F
  output = F.relu(input)
  ```

  `nn.Module` 适用于需要可学习参数的层（如 `nn.Linear`、`nn.Conv2d`），而 `F` 适用于无状态操作（如 `relu`、`softmax`）。

------

## 2. 常用 API 及其用法

### 2.1 激活函数（Activation Functions）

#### ReLU（Rectified Linear Unit）

```python
import torch.nn.functional as F
import torch

tensor = torch.tensor([-1.0, 0.0, 1.0])
output = F.relu(tensor)
print(output)  # tensor([0., 0., 1.])
```

#### Softmax

```python
x = torch.randn(3)
output = F.softmax(x, dim=0)
print(output)
```

> **注意：** `dim` 指定沿哪个维度进行 softmax 计算。

### 2.2 归一化（Normalization）

#### Batch Normalization（批归一化）

```python
x = torch.randn(10, 5)  # 10个样本，每个5维
running_mean = torch.zeros(5)
running_var = torch.ones(5)
output = F.batch_norm(x, running_mean, running_var, training=True)
```

#### Layer Normalization（层归一化）

```python
x = torch.randn(10, 5)
norm_shape = (5,)
output = F.layer_norm(x, norm_shape)
```

### 2.3 卷积（Convolution）

#### `F.conv2d`

```python
input = torch.randn(1, 3, 32, 32)  # NCHW
weight = torch.randn(6, 3, 5, 5)  # out_channels, in_channels, kernel_size, kernel_size
output = F.conv2d(input, weight, bias=None, stride=1, padding=0)
```

> **注意：** `torch.nn.Conv2d` 具有可学习参数，而 `F.conv2d` 需要显式提供 `weight`。

### 2.4 池化（Pooling）

#### `F.max_pool2d`

```python
input = torch.randn(1, 3, 32, 32)
output = F.max_pool2d(input, kernel_size=2, stride=2)
```

#### `F.avg_pool2d`

```python
output = F.avg_pool2d(input, kernel_size=2, stride=2)
```

### 2.5 损失函数（Loss Functions）

#### Cross Entropy Loss（交叉熵损失）

```python
y_pred = torch.randn(3, 5)  # 3个样本，5个类别
y_true = torch.tensor([1, 0, 3])  # 真实类别
loss = F.cross_entropy(y_pred, y_true)
```

> **注意：** `F.cross_entropy` 相当于 `log_softmax + nll_loss`。

#### Mean Squared Error Loss（均方误差）

```python
output = F.mse_loss(torch.randn(3), torch.randn(3))
```

### 2.6 Dropout（随机失活）

```python
x = torch.randn(5, 5)
output = F.dropout(x, p=0.5, training=True)
```

> **注意：** 在 `eval()` 模式下，`dropout` 不会生效。

------

## 3. 使用注意事项

1. **可微分性**：
   - `F` 模块中的大多数操作都支持自动求导，因此可直接用于计算梯度。
2. **`F` 模块 vs. `nn.Module`**：
   - `torch.nn` 层通常包含 `state`（如 `BatchNorm` 存储均值/方差），而 `F` 需要手动传递这些参数。
3. **`F.softmax` vs. `F.log_softmax`**：
   - 在交叉熵损失计算时，建议使用 `F.log_softmax + F.nll_loss` 以提高数值稳定性。
4. **参数共享**：
   - `F.conv2d` 允许在多个 `forward` 过程中使用相同的权重。

------

## 4. 参考文档

- PyTorch 官方文档：https://pytorch.org/docs/stable/nn.functional.html
- 深入理解 PyTorch F 模块：https://pytorch.org/tutorials

以上内容涵盖了 `torch.nn.functional` 的基础知识、常见 API 及使用示例，希望对你的学习有所帮助！