在 PyTorch 中，`nn.Linear` 是一个全连接层，它包含可学习的权重（`weight`）和偏置（`bias`）。如果你想查看一个 `nn.Linear` 层的权重系数，可以通过访问其 `weight` 属性来实现。

# 示例代码
以下是一个完整的示例，展示如何定义一个 `nn.Linear` 层并查看其权重和偏置：

```python
import torch
import torch.nn as nn

# 定义一个全连接层
fc = nn.Linear(10, 5)  # 输入特征维度为 10，输出维度为 5

# 查看权重
print("Weights shape:", fc.weight.shape)
print("Weights:", fc.weight)

# 查看偏置
print("Bias shape:", fc.bias.shape)
print("Bias:", fc.bias)
```

# 输出示例
假设权重和偏置已经初始化，输出可能如下：
```plaintext
Weights shape: torch.Size([5, 10])
Weights: tensor([[ 0.1234, -0.5678,  0.9012, -0.3456,  0.7890, -0.1111,  0.2222, -0.3333,  0.4444, -0.5555],
                 [ 0.6666, -0.7777,  0.8888, -0.9999,  0.1111, -0.2222,  0.3333, -0.4444,  0.5555, -0.6666],
                 [ 0.7777, -0.8888,  0.9999, -0.1111,  0.2222, -0.3333,  0.4444, -0.5555,  0.6666, -0.7777],
                 [ 0.8888, -0.9999,  0.1111, -0.2222,  0.3333, -0.4444,  0.5555, -0.6666,  0.7777, -0.8888],
                 [ 0.9999, -0.1111,  0.2222, -0.3333,  0.4444, -0.5555,  0.6666, -0.7777,  0.8888, -0.9999]])

Bias shape: torch.Size([5])
Bias: tensor([-0.1234,  0.5678, -0.9012,  0.3456, -0.7890])
```

# 关键点解释
1. **权重（`weight`）**：
   - 形状为 `[out_features, in_features]`，即 `[5, 10]`。
   - 每一行代表一个输出特征的权重。
   - 每一列对应一个输入特征的权重。

2. **偏置（`bias`）**：
   - 形状为 `[out_features]`，即 `[5]`。
   - 每个偏置值对应一个输出特征。

3. **初始化**：
   - 默认情况下，`nn.Linear` 的权重和偏置会自动初始化。权重通常使用 Kaiming 初始化，偏置初始化为 0。
   - 如果需要自定义初始化，可以通过以下方式：
     ```python
     nn.init.xavier_uniform_(fc.weight)
     nn.init.constant_(fc.bias, 0.1)
     ```

# 总结
通过访问 `fc.weight` 和 `fc.bias`，你可以查看 `nn.Linear` 层的权重和偏置。这些参数是可学习的，会在训练过程中通过反向传播进行更新。




[[He Initialization]]
**Kaiming 初始化（He 初始化）**确实是随机的，但它服从特定的分布，这些分布是经过精心设计的，以确保网络在训练过程中能够更好地传播信号并避免梯度消失或爆炸问题。

### Kaiming 初始化的随机性
Kaiming 初始化的核心思想是为权重选择一个合适的随机分布，使得网络的输入和输出在正向传播和反向传播时保持稳定的方差。具体来说，权重的初始化值是随机的，但这些随机值服从特定的分布，例如：

1. **Kaiming Normal（正态分布）**：
   - 权重从均值为 0、标准差为$\sqrt{\frac{2}{\text{fan\_in}}}$ 的正态分布中随机采样。
   - 公式：  
     $$
     W \sim \mathcal{N}\left(0, \frac{2}{\text{fan\_in}}\right)
     $$
     
   
2. **Kaiming Uniform（均匀分布）**：
   
   - 权重从范围为$\left[-\sqrt{\frac{6}{\text{fan\_in}}}, \sqrt{\frac{6}{\text{fan\_in}}}\right]$的均匀分布中随机采样。
   - 公式： 
     $$
     W \sim \mathcal{U}\left(-\sqrt{\frac{6}{\text{fan\_in}}}, \sqrt{\frac{6}{\text{fan\_in}}}\right)
     $$
     

### 随机性的作用
- **随机初始化**：随机性确保了每个权重的初始值是独立的，避免了所有权重相同的情况。如果所有权重初始化为相同的值，网络在训练过程中可能会陷入对称性问题，导致梯度更新完全一致，无法学习到有用的特征。
- **特定分布**：虽然权重是随机的，但它们服从特定的分布，这些分布的参数（如方差）是根据网络的结构（如输入神经元数量 $\text{fan\_in}$）和激活函数（如 ReLU）精心设计的，以确保网络在训练初期能够稳定地传播信号。

### 示例代码
以下是一个简单的 PyTorch 示例，展示如何使用 Kaiming 初始化：

```python
import torch
import torch.nn.init as init

# 创建一个权重张量
weight = torch.empty(5, 10)  # 输出维度为 5，输入维度为 10

# 使用 Kaiming Normal 初始化
init.kaiming_normal_(weight, mode='fan_in', nonlinearity='relu')
print("Kaiming Normal Initialization:\n", weight)

# 创建另一个权重张量
weight_uniform = torch.empty(5, 10)

# 使用 Kaiming Uniform 初始化
init.kaiming_uniform_(weight_uniform, mode='fan_in', nonlinearity='relu')
print("\nKaiming Uniform Initialization:\n", weight_uniform)
```

### 输出示例
每次运行代码时，权重的值都会不同，因为它们是从随机分布中采样的。例如：
```plaintext
Kaiming Normal Initialization:
 tensor([[ 0.3478, -0.2345,  0.1234, ...,  0.4567, -0.5678,  0.6789],
         [-0.7890,  0.8901, -0.9012, ..., -0.1234,  0.2345, -0.3456],
         [ 0.4567, -0.5678,  0.6789, ...,  0.7890, -0.8901,  0.9012],
         [-0.1234,  0.2345, -0.3456, ..., -0.4567,  0.5678, -0.6789],
         [ 0.7890, -0.8901,  0.9012, ..., -0.1234,  0.2345, -0.3456]])

Kaiming Uniform Initialization:
 tensor([[ 0.1234, -0.2345,  0.3456, ...,  0.4567, -0.5678,  0.6789],
         [-0.7890,  0.8901, -0.9012, ..., -0.1234,  0.2345, -0.3456],
         [ 0.4567, -0.5678,  0.6789, ...,  0.7890, -0.8901,  0.9012],
         [-0.1234,  0.2345, -0.3456, ..., -0.4567,  0.5678, -0.6789],
         [ 0.7890, -0.8901,  0.9012, ..., -0.1234,  0.2345, -0.3456]])
```

### 总结
Kaiming 初始化是随机的，但这种随机性是经过精心设计的，权重从特定的分布中采样，以确保网络在训练初期能够稳定地传播信号。这种初始化方法特别适用于使用 ReLU 激活函数的深度神经网络，能够有效缓解梯度消失和爆炸问题。