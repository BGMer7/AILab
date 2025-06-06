[[Auto Regressive(AR)]]
[[Deep-Learning/Transformer]]
[[Non-Autoregressive Transformer(NAT)]]

### **自回归 Transformer (AT)**
自回归 Transformer 生成序列时遵循自回归模式，即当前的输出依赖于前面已经生成的输出。

#### **特点**
1. **逐步生成**：每一步都基于前面的生成结果进行预测。
2. **生成公式**：
   $$
   P(Y) = \prod_{t=1}^T P(y_t | y_1, y_2, \dots, y_{t-1}, X)
   $$
   其中 $Y = \{y_1, y_2, \dots, y_T\}$ 是目标序列，$X$ 是输入序列。
3. **模型结构**：通常由编码器-解码器 Transformer 组成。编码器处理输入序列，解码器通过掩码注意力逐步生成输出。
4. **优点**：
   - 生成质量高。
   - 序列间依赖建模强。
5. **缺点**：
   - **生成速度慢**：因为生成是串行的，后一个 token 的生成依赖于前一个 token。
   - 对长序列的生成可能出现错误传播问题。

#### **典型模型**
- **Transformer 原始版本**（Vaswani 等，2017）
- **GPT 系列**：用于语言生成。

#### **应用场景**
- 机器翻译（如 seq2seq）
- 文本生成（如语言建模）
