# PyTorch `torch.nn.ModuleList` 详解

## 1. `torch.nn.ModuleList` 简介

`torch.nn.ModuleList` 是 `torch.nn.Module` 的一个子类，用于存储多个 `nn.Module` 子模块的 **有序列表**。与 Python 的 `list` 不同，`ModuleList` 会自动注册其中的子模块，使其能够被 `parameters()` 方法正确管理。

### **主要特点：**

- 自动注册子模块：

  `ModuleList` 里面的子模块会被 `model.parameters()` 追踪，并正确进行梯度计算。

- 类似 Python `list`：

  `ModuleList` 支持索引、迭代、追加等操作，但不会执行前向传播（需要显式调用子模块）。

- 不等同于 `Sequential`：

  `Sequential` 直接执行前向传播，而 `ModuleList` 只是存储子模块，需要手动调用。

------

## 2. `ModuleList` API 及用法

### **2.1 创建 `ModuleList`**

```python
import torch.nn as nn

layers = nn.ModuleList([
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
])
```

> **注意：** 这里 `ModuleList` 只是存储了 `Linear` 和 `ReLU`，并不会自动执行前向传播。
>
> 直接调用 `model(input)` **无法得到计算结果**，因为 `ModuleList` 只是一个**存储子模块的容器**，它本身**没有 forward 方法**。如果直接用 `model(input)`，会报错。

### **2.2 访问 `ModuleList` 中的子模块**

```python
print(layers[0])  # nn.Linear(10, 20)
print(layers[1])  # nn.ReLU()
```

### **2.3 迭代 `ModuleList`**

```python
for layer in layers:
    print(layer)
```

### **2.4 动态添加子模块**

```python
layers.append(nn.Linear(5, 2))
print(len(layers))  # 4
```

> **注意：** `ModuleList.append()` 允许动态添加子模块，但不能像 `list` 那样存储普通 `tensor` 或 `str`。

### **2.5 在 `forward()` 方法中使用 `ModuleList`**

```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        ])

    def forward(self, x):
        for layer in self.layers:
            # 每个模块单独前向传播
            x = layer(x)
        return x

model = MyModel()
print(model)
```

> **注意：** `ModuleList` 只是存储子模块，`forward()` 里仍需手动遍历并执行计算。

------

## 3. `ModuleList` vs. `Sequential`

`ModuleList` 和 `Sequential` 都是 `torch.nn.Module` 的容器，但它们在**用法**和**功能**上存在明显区别：

| 特性                 | `ModuleList`                                           | `Sequential`                                 |
| -------------------- | ------------------------------------------------------ | -------------------------------------------- |
| 是否自动执行前向传播 | ❌ 否，需要手动遍历并调用每个子模块                     | ✅ 是，直接调用 `Sequential` 即可执行前向传播 |
| 是否能动态添加层     | ✅ 是，可用 `.append()` 添加新层                        | ❌ 否，初始化后不能修改                       |
| 是否支持索引访问     | ✅ 是，可以 `module_list[i]` 访问某个子模块             | ✅ 是，可以 `sequential[i]` 访问某个子模块    |
| 是否支持迭代访问     | ✅ 是，可以 `for layer in module_list:` 迭代            | ✅ 是，可以 `for layer in sequential:` 迭代   |
| 适用于               | **非线性结构**（例如 ResNet 残差块、Transformer 结构） | **线性结构**（例如前馈神经网络、简单 CNN）   |

------

### **示例对比**

#### **1. `ModuleList` 需要手动调用子模块**

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)  # 需要手动调用每一层
        return x

model = MyModel()
```

> **适用于** 需要动态添加层或自定义前向传播逻辑的情况。

------

#### **2. `Sequential` 直接执行前向传播**

```python
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

output = model(torch.randn(1, 10))  # 直接传入数据即可
```

> **适用于** 层按顺序执行、无需动态修改的情况（如常规 MLP）。

------

### **总结**

- **如果层的顺序固定，并且只需按顺序执行前向传播**，`Sequential` 更简洁高效。
- **如果需要动态调整层、灵活控制前向传播（如跳跃连接、残差连接）**，`ModuleList` 更合适。

------

## 4. `ModuleList` 进阶用法

### **4.1 生成多个相同层**

```python
num_layers = 3
hidden_dim = 20

model = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
```

### **4.2 结合 `ModuleDict` 进行结构化存储**

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleDict({
            'encoder': nn.ModuleList([nn.Linear(10, 10) for _ in range(3)]),
            'decoder': nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
        })
    
    def forward(self, x):
        for layer in self.blocks['encoder']:
            x = layer(x)
        for layer in self.blocks['decoder']:
            x = layer(x)
        return x
```

> `ModuleDict` 让 `ModuleList` 更具可读性和层级结构。

------

## 5. 使用 `ModuleList` 需注意的事项

1. **不要直接传递普通 `list`**

   ```python
   self.layers = [nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5)]  # 错误！
   ```

   > 这样 `self.layers` 不会被 `parameters()` 追踪，应使用 `nn.ModuleList`。

2. **`ModuleList` 仅存储，不执行 forward**

   ```python
   x = torch.randn(1, 10)
   model = nn.ModuleList([nn.Linear(10, 20), nn.ReLU()])
   output = model(x)  # 错误！
   ```

   > 需要显式调用 `for layer in model:` 遍历并执行计算。

3. **动态网络构建**

   - `ModuleList` 适用于逐层构建网络（如 Transformer、RNN 变体）。
   - 若网络结构固定，建议使用 `Sequential`。

------

## 6. 参考文档

- PyTorch 官方文档: https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
- PyTorch 教程: https://pytorch.org/tutorials/