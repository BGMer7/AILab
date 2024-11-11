[[Batch Norm]]
[[Normalization]]

Transformer中的归一化(五)：Layer Norm的原理和实现 & 为什么Transformer要用LayerNorm - Gordon Lee的文章 - 知乎
https://zhuanlan.zhihu.com/p/492803886

Layer Normalization（层归一化）是一种在神经网络中对输入层进行归一化的技术，常用于自然语言处理（NLP）和计算机视觉（CV）模型。与 Batch Normalization（批归一化）不同的是，Layer Normalization 是对每一个样本的特征维度进行归一化，而不是跨样本批次进行归一化。

## Layer Normalization 的工作原理
1. **计算均值和方差**：对每个输入样本的特征维度计算均值和方差。
2. **归一化**：用均值减去特征值，再除以方差的平方根，加上一个很小的 `epsilon` 值（防止除零），从而得到归一化后的输出。
3. **尺度和平移**：通常引入可训练的缩放参数 `gamma` 和偏移参数 `beta`，使模型能够学习最优的归一化。



### 优点

- **适用于小批次或变动批次大小的输入**：由于它是对单个样本的特征进行归一化，Layer Normalization 不依赖于批量大小，因此适合小批量甚至单样本的处理。
- **稳定性**：在梯度下降过程中提供稳定的梯度，有助于提升模型训练速度和收敛性能。

Layer Normalization 通常用于循环神经网络（RNNs）和 Transformers 等架构，在处理每个时间步的输出时尤为有效。

总体来说，BN更适合CV的任务，而LN更适合NLP的任务。



## 实现

以下是一个使用 PyTorch 实现 Layer Normalization 的简单示例：

```python
import torch
import torch.nn as nn

# 创建一个具有随机值的输入张量 (batch_size, num_features)
input_tensor = torch.randn(3, 5)  # 假设 batch_size=3, num_features=5

# 创建 LayerNorm 实例，指定特征数
layer_norm = nn.LayerNorm(normalized_shape=5)  # normalized_shape 对应特征维度的大小

# 进行前向传播，得到归一化后的输出
output = layer_norm(input_tensor)

print("Input Tensor:")
print(input_tensor)
print("\nLayer Normalized Output:")
print(output)
```

### 解释
- `normalized_shape=5`：指定要归一化的特征数，这里设为 `5`，因为输入张量的每一行包含 5 个特征。
- `nn.LayerNorm`：PyTorch 中的 `LayerNorm` 类会对每个样本的特征维度进行归一化操作。

### 结果
这个代码会输出原始的输入张量和经过层归一化后的输出。输出的每一行（对应一个样本）的均值为 0，标准差为 1，从而实现了归一化。

### 在模型中的应用
Layer Normalization 通常在模型的层级定义中使用，以下是一个包含 Layer Normalization 的简单神经网络示例：

```python
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.layer_norm(x)
        return x

# 使用该模型
input_tensor = torch.randn(3, 5)  # 输入张量 (batch_size, num_features)
model = SimpleModel(input_dim=5)
output = model(input_tensor)

print("Model Output with Layer Normalization:")
print(output)
```

这个 `SimpleModel` 会先通过线性层（`fc`）处理输入数据，然后进行 Layer Normalization。





![img](https://pic4.zhimg.com/v2-6d444305489675100aef96293cc3a34d_r.jpg)

> 一言以蔽之。**BN是对batch的维度去做归一化，也就是针对不同样本的同一特征做操作。LN是对hidden的维度去做归一化，也就是针对单个样本的不同特征做操作。**因此**LN可以不受样本数的限制。**
>
> 具体而言**，BN就是在每个维度上统计所有样本的值，计算均值和方差；LN就是在每个样本上统计所有维度的值，计算均值和方差**（注意，这里都是指的简单的MLP情况，输入特征是（bsz，hidden_dim））。所以BN在每个维度上分布是稳定的，LN是每个样本的分布是稳定的。

