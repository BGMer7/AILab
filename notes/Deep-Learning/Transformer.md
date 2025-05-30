[[Self-Attention]]


# Component

## Feed-Forward Network (FFN)

或者称为 FFNN (Feed-forward Neural Network)



在Transformer模型中，每个编码器和解码器层都包括两个主要部分：多头自注意力机制（Multi-Head Self-Attention）和前馈神经网络（Feed Forward Neural Network，FFN）。FFN通常由两个线性变换和一个激活函数组成，细节如下：

- **输入**：来自自注意力机制的输出。
- **线性变换**：首先应用一个线性变换（全连接层），将输入映射到一个高维空间。
- **激活函数**：接着使用非线性激活函数，通常是ReLU（Rectified Linear Unit）。
- **再次线性变换**：最后，再次通过一个线性变换将数据映射回原来的维度。



FFN的本质就是一个两层的MLP。

这个MLP的简单数学表示是：
$$
FFN(x) = f(x \cdot W_1^T) \cdot W_2
$$
在这两层MLP中，第一层将输入的向量升维，第二层将向量重新降维，这样可以学习到更加抽象的特征。

### 主要作用

FFN的主要作用是在输入的特征上进行非线性变换，从而增加模型的表达能力。自注意力机制主要负责捕捉输入序列中各个元素之间的关系（上下文信息），而FFN通过引入非线性特性，使得模型能够更好地学习复杂的表示。

在自然语言处理中，语言的复杂性往往不仅仅依赖于词与词之间的关系，还涉及上下文的多样性和丰富性。FFN通过激活函数引入了非线性，可以帮助模型更好地处理这些复杂的关系。此外，FFN也为每个位置的表示引入了独立的学习能力，使得模型可以处理不同位置的模式。

### FFN 和 Transformer

在Transformer模型中，FFN位于多头自注意力机制之后。每个编码器和解码器层都包含一个FFN，用于对自注意力机制的输出进行进一步的处理。这种设计使得模型能够在捕获全局依赖关系的同时，对局部特征进行更精细的处理。

## Multi-Layer Perceptron (MLP)

MLP（Multi-Layer Perceptron）和全连接层（Fully Connected Layer，也称为 Dense Layer）之间有密切的关系，但它们并不完全相同。MLP 是一种神经网络架构，而全连接层是构成 MLP 的基本组件之一。以下是它们之间的关系和区别：

### MLP

1. **定义**：
   - MLP 是一种前馈神经网络（Feed-Forward Neural Network），由多个层组成，每层包含多个神经元（节点）。MLP 至少包含三层：输入层、一个或多个隐藏层和输出层。
   - MLP 的关键特点是每层的神经元与下一层的神经元完全连接，即每个神经元都与下一层的所有神经元相连。
2. **结构**：
   - **输入层**：接收输入数据。
   - **隐藏层**：对输入数据进行非线性变换，提取特征。
   - **输出层**：生成最终的输出结果。
3. **激活函数**：
   - MLP 中的每个神经元通常使用非线性激活函数（如 ReLU、Sigmoid 或 Tanh）来引入非线性特性，使网络能够学习复杂的模式。
4. **用途**：
   - MLP 是一种通用的神经网络架构，可以用于各种任务，包括分类、回归和特征提取等。

### Dense Layer

1. **定义**：

   - 全连接层是神经网络中的一种层类型，其中每个神经元与前一层的所有神经元相连。全连接层通常用于 MLP 中，也可以用于其他神经网络架构（如 CNN 和 RNN）中。

2. **结构**：

   - 全连接层由一组权重（参数）和偏置组成。每个神经元的输出是前一层所有神经元输出的加权和加上一个偏置，然后通过激活函数进行非线性变换。

3. **数学表示**：

   - 假设输入向量为 $x∈Rn$，权重矩阵为 $W∈Rm×n$，偏置向量为 $b∈Rm$，则全连接层的输出为：

     $y=f(xW+b)$

     其中，$f$是激活函数。

4. **用途**：

   - 全连接层通常用于神经网络的最后几层，将特征映射到输出空间。在 MLP 中，全连接层是构成网络的主要组件。

### 关系和区别

- **关系**：
  - MLP 是一种神经网络架构，由多个全连接层组成。
  - 全连接层是 MLP 的基本组件，用于实现层与层之间的连接和数据变换。
- **区别**：
  - **MLP** 是一种网络架构，包含多个层（输入层、隐藏层和输出层），每层之间通过全连接层连接。
  - **全连接层** 是一种层类型，可以用于 MLP 中，也可以用于其他神经网络架构中。全连接层负责实现层与层之间的连接和数据变换。

MLP 是一种由多个全连接层组成的神经网络架构，而全连接层是构成 MLP 的基本组件。

全连接层负责实现层与层之间的连接和数据变换，而 MLP 则通过多个全连接层的堆叠来实现复杂的特征提取和模式学习。



## Multi Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        
        self.h = heads
        self.d_k = d_model // heads
        self.d_model = d_model
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        scores = F.softmax(scores, dim=-1)
        
        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        return output
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # 矩阵转置
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        
        # 拼接多头注意力输出
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)

        return output
    
```

多头注意力机制（Multi-Head Attention）之所以能**提取不同子空间的特征**，核心原因可以归结为：

------

### 每个头有**独立的线性变换参数**

在 Transformer 的多头注意力中，每个注意力头都有 **自己的投影矩阵**：

```python
Q_i = X @ W_Q_i  
K_i = X @ W_K_i  
V_i = X @ W_V_i  
```

- `W_Q_i`, `W_K_i`, `W_V_i` 是第 `i` 个注意力头独立学习的线性变换参数；
- 不同的头看到的是同一个输入 `X`，但它们的视角不同 —— 由 `W_*` 决定；
- 就像多个摄像头拍摄同一个物体，但从不同角度、光线、滤镜来看，看到的细节不同。

------

### 换句话说：**多头 = 多种特征提取方式**

比如：

- 某个注意力头可能学会了“关注主语与动词之间的关系”；
- 另一个注意力头可能学会了“捕捉位置之间的依赖关系”；
- 有的关注句法结构，有的关注实体之间的距离等。

**这些都是在训练中自然学出来的。**

论文《Attention is All You Need》中就指出，多头注意力的确能学到**不同的语义模式**。后续的可视化研究（如 BERT 可视化工具）也发现：

- 某些头专注于对称结构；
- 某些头聚焦长距离依赖；
- 某些头高度专注于特殊 token（如 `[CLS]`、句首等）；

这些差异性的出现正是因为每个头有**自己独立的投影子空间**。



> 多头注意力能提取不同子空间特征，是因为每个头都在用**不同的线性变换矩阵**从输入中投影出**不同的子表示（子空间）**，从而在训练中学会关注输入数据的**不同语义维度和模式**。

如果你想，我可以给你画个图或者用代码可视化不同注意力头的关注行为。

### mask

```python
def test_with_mask():
    """测试掩码功能"""
    print("\n🎭 测试掩码功能")
    print("="*50)
    
    batch_size = 1
    seq_length = 5
    d_model = 128
    heads = 4
    
    mha = MultiHeadAttention(heads=heads, d_model=d_model)
    input_data = torch.randn(batch_size, seq_length, d_model)
    
    # 创建掩码：假设最后两个位置是padding
    mask = torch.ones(batch_size, seq_length)
    mask[0, 3:] = 0  # 掩盖最后两个位置
    
    print(f"📋 掩码模式: {mask[0].tolist()}")
    print("   (1表示有效位置，0表示padding位置)")
    
    with torch.no_grad():
        # 不使用掩码
        output_no_mask = mha(input_data, input_data, input_data)
        
        # 使用掩码
        output_with_mask = mha(input_data, input_data, input_data, mask=mask)
        
        print(f"\n📊 掩码效果对比:")
        for i in range(seq_length):
            no_mask_norm = torch.norm(output_no_mask[0, i])
            with_mask_norm = torch.norm(output_with_mask[0, i])
            mask_status = "有效" if mask[0, i] == 1 else "掩盖"
            print(f"   位置{i+1}({mask_status}): 无掩码={no_mask_norm:.3f}, 有掩码={with_mask_norm:.3f}")
```

输出结果：

```python
🎭 测试掩码功能
==================================================
📋 掩码模式: [1.0, 1.0, 1.0, 0.0, 0.0]
   (1表示有效位置，0表示padding位置)

📊 掩码效果对比:
   位置1(有效): 无掩码=1.775, 有掩码=2.190
   位置2(有效): 无掩码=1.713, 有掩码=2.046
   位置3(有效): 无掩码=1.875, 有掩码=2.087
   位置4(掩盖): 无掩码=1.785, 有掩码=2.111
   位置5(掩盖): 无掩码=1.981, 有掩码=2.289
```

#### 🔍 为什么用 `torch.norm()` 来对比注意力输出结果？

##### 1. **`torch.norm` 是在衡量一个向量的整体“大小”或“能量”**

对于 Transformer 中的注意力输出，每个位置 `output[i]` 是一个向量（维度为 `d_model`）。我们可以认为它代表了该位置在上下文中的综合语义。

- `torch.norm(output[i])` ≈ 向量在所有维度上的强度（欧几里得范数）
- 它压缩了高维向量为一个标量，**便于做对比和可视化**

------

##### 2. **用于观察 mask 是否成功起作用**

你的测试对比的是：

```python
output_no_mask vs output_with_mask
```

用 `norm` 比较每个位置的变化：

- ✅ 如果是有效位置（mask = 1），那么 `output_with_mask[i]` 应该与 `output_no_mask[i]` **差不多**；
- 🚫 如果是被掩盖的位置（mask = 0），那么 `output_with_mask[i]` 会明显受到注意力屏蔽的影响，**norm 会大幅下降或改变**。

这就是通过 norm 对比，来验证 mask 机制有没有起作用的核心目的。

------

##### 3. **避免直接打印向量**

- `d_model = 128`，打印完整向量会非常冗长，难以直接看出差异；
- 用 norm 把每个向量变成一个标量，更容易一眼看出差异。

你使用 `torch.norm()` 来比较掩码前后的注意力输出，是为了：

| 原因       | 解释                                                |
| ---------- | --------------------------------------------------- |
| 语义代表性 | 注意力输出是一个高维语义向量，norm 能反映其强度变化 |
| 数值简洁   | 比较标量更直观，避免高维向量冗长展示                |
| 验证效果   | 可用于判断 mask 是否对被掩盖位置起了“屏蔽”作用      |

所以这是一个**实用且简洁有效的调试方法**。你可以进一步加上一些 `cosine similarity` 或 `L2 diff` 来更细粒度地比较两个输出的变化。需要的话我可以帮你加上这些分析。
