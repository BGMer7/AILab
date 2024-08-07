### 神经网络的反向传播

在神经网络的训练过程中，正向传播（forward propagation）和反向传播（backpropagation）是两个关键步骤。正向传播计算每一层的输出，直到最终输出层产生预测值；反向传播则计算损失函数相对于每个权重的梯度，并更新权重以最小化损失函数。

#### 1. 正向传播（Forward Propagation）

在正向传播过程中，输入数据依次通过网络的每一层，计算并记录每一层的激活值和未激活值（即线性组合的结果）。具体来说，对于每一层 $ l $：

- 输入 $ a^{[l-1]} $
- 线性组合 $z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]} $
- 激活函数 $a^{[l]} = \sigma(z^{[l]}) $

这些值需要在内存中记录下来，以便在反向传播时使用。

#### 2. 反向传播（Backpropagation）

在反向传播过程中，从输出层开始，计算损失函数对每个参数的偏导数，并依次向前传播到每一层。具体步骤如下：

1. **计算输出层的误差**：对于输出层，计算损失函数 $ L $ 对输出 $ a^{[L]} $ 的偏导数，即 $\frac{\partial L}{\partial a^{[L]}} $。

2. **传播误差到每一层**：对于每一层 $l$（从后向前）计算：
   - $\delta^{[l]} = \frac{\partial L}{\partial z^{[l]}} = \frac{\partial L}{\partial a^{[l]}} \cdot \sigma'(z^{[l]}) $
   - 更新权重和偏置：
     - $\frac{\partial L}{\partial W^{[l]}} = \delta^{[l]} \cdot (a^{[l-1]})^T $
     - $\frac{\partial L}{\partial b^{[l]}} = \delta^{[l]} $

这些计算需要依赖于前向传播过程中记录的 $z^{[l]} $ 和 $ a^{[l]} $ 值。

### 需要记录的值

为了实现有效的反向传播，确实需要在内存中记录每一层的以下值：

1. **激活值 $ a^{[l]} $**：记录每一层的输出，作为下一层的输入，以及用于反向传播计算梯度。
2. **未激活值 $z^{[l]} $**：记录每一层的线性组合结果，用于计算激活函数的导数。
3. **误差值 $\delta^{[l]} $**：在反向传播过程中计算并记录每一层的误差，用于更新权重和偏置。

### 反向传播示例

假设一个简单的三层神经网络，层与层之间使用 sigmoid 激活函数。前向传播和反向传播过程如下：

**前向传播**：
1. 输入层到隐藏层：
   $$
   z^{[1]} = W^{[1]} x + b^{[1]}
   $$
   $$
   a^{[1]} = \sigma(z^{[1]})
   $$

2. 隐藏层到输出层：
   $$
   z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}
   $$
   $$
   a^{[2]} = \sigma(z^{[2]})
   $$

**反向传播**：
1. 输出层误差：
   $$
   \delta^{[2]} = \frac{\partial L}{\partial a^{[2]}} \cdot \sigma'(z^{[2]})
   $$

2. 隐藏层误差：
   $$
   \delta^{[1]} = (W^{[2]})^T \delta^{[2]} \cdot \sigma'(z^{[1]})
   $$

3. 更新权重和偏置：
   $$
   \frac{\partial L}{\partial W^{[2]}} = \delta^{[2]} \cdot (a^{[1]})^T
   $$
   $$
   \frac{\partial L}{\partial b^{[2]}} = \delta^{[2]}
   $$
   $$
   \frac{\partial L}{\partial W^{[1]}} = \delta^{[1]} \cdot x^T
   $$
   $$
   \frac{\partial L}{\partial b^{[1]}} = \delta^{[1]}
   $$
