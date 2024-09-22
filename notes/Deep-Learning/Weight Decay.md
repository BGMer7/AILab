# Weight Decay

**Weight Decay（权重衰减）** 并不属于  **Learning Rate（学习率）** 的一部分，但它们密切相关，且经常在一起使用。下面是对它们之间关系的详细解释：

## Weight Decay

Weight Decay 是一种正则化方法，**通常用于防止模型的过拟合**。

在损失函数中，weight decay是放在正则项（regularization）前面的一个系数，正则项一般指示模型的复杂度，所以weight decay的作用是调节模型复杂度对损失函数的影响，若weight decay很大，则复杂的模型损失函数的值也就大。

它通过在损失函数中加入一个权重参数的平方和的惩罚项，来限制权重的大小，从而防止权重过大导致模型复杂度过高。

如果损失函数是 $ L(\theta) $，那么使用 Weight Decay 后的损失函数变为：

$$
L_{total}(\theta) = L(\theta) + \lambda ||\theta||^2
$$

其中：

- $ L(\theta) $ 是原始损失函数（如交叉熵损失或均方误差）。
- $ \lambda $ 是权重衰减系数（正则化强度），$\lambda$ 越大，正则化强度越大，可以理解为权重越接近0，模型的泛化能力越好。
- $ ||\theta||^2 $ 是模型权重的平方和，本质是一个$L2$ 正则化系数。

这个惩罚项可以迫使模型的权重较小，避免模型过于复杂。



### Function

使用 weight decay 可以：

- 防止[过拟合](https://zhida.zhihu.com/search?q=%E8%BF%87%E6%8B%9F%E5%90%88&zhida_source=entity&is_preview=1)

- 保持权重在一个较小在的值，避免梯度爆炸。

- 因为在原本的 loss 函数上加上了权重值的 L2 [范数](https://zhida.zhihu.com/search?q=%E8%8C%83%E6%95%B0&zhida_source=entity&is_preview=1)，在每次迭代时，模型不仅会去**优化/最小化 loss**，还会使**模型权重最小化**。

- 让权重值保持尽可能小，有利于**控制权重值的变化幅度**(如果梯度很大，说明模型本身在变化很大，去过拟合样本)，从而**避免梯度爆炸**。

### Implements



**Weight Decay（权重衰减）** 有几种不同的实现形式，主要用于限制模型权重的增长，防止过拟合。以下是几种常见的形式：

#### L2 Regularization

这是最常见的 Weight Decay 形式，也称为 **L2范数**。它通过将权重平方的惩罚项添加到损失函数中，迫使权重保持较小。损失函数变为：

$$
L_{total}(\theta) = L(\theta) + \lambda ||\theta||^2_2
$$




其中：

- $L(\theta)$ 是原始损失函数。
- $||\theta||^2_2$ 是所有权重平方和的二范数。
- $\lambda$ 是正则化强度（权重衰减系数）。

在反向传播过程中，这会在更新权重时添加一个与权重本身成比例的惩罚项，从而减小权重。

**更新公式**：

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\partial L(\theta)}{\partial \theta} - \eta \cdot \lambda \theta_t
$$




其中，$\eta$ 是学习率，\$lambda\$是权重衰减系数。

L2 正则化也是神经网络优化器（如 SGD）中最常用的 Weight Decay 形式。

#### L1 Regularization

**L1 正则化**是另一种形式的 Weight Decay，它通过惩罚权重的绝对值而不是平方值。损失函数变为：

$$
L_{total}(\theta) = L(\theta) + \lambda ||\theta||_1
$$




其中：

- $||\theta||_1$ 是权重的绝对值和（L1范数）。

L1 正则化的主要特点是它能促使部分权重变为 0，从而产生一个稀疏的权重矩阵，这在一些稀疏模型中比较有用（如 LASSO 回归）。

**更新公式**：

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\partial L(\theta)}{\partial \theta} - \eta \cdot \lambda \cdot \text{sign}(\theta_t)
$$




其中 $\text{sign}(\theta_t)$ 是权重的符号函数。

#### Elastic Net Regularization

**Elastic Net 正则化**结合了 L1 和 L2 正则化，将两者的优势结合在一起。损失函数为：

$$
L_{total}(\theta) = L(\theta) + \lambda_1 ||\theta||_1 + \lambda_2 ||\theta||^2_2
$$

- $\lambda_1$ 是控制 L1 正则化强度的超参数。
- $\lambda_2$ 是控制 L2 正则化强度的超参数。

Elastic Net 正则化既能促使部分权重稀疏（通过 L1），又能防止过拟合（通过 L2）。这种方式在高维特征和稀疏性要求的场景中使用较多。

#### Decoupled Weight Decay

解耦权重衰减

在一些现代优化算法中（如 AdamW），Weight Decay 与 L2 正则化被解耦。传统的 Weight Decay 会直接与损失函数的梯度结合，但在解耦的权重衰减中，权重衰减独立于梯度更新执行。

AdamW 的更新规则：

$$
\theta_{t+1} = \theta_t - \eta \cdot (\frac{\partial L(\theta)}{\partial \theta} + \lambda \cdot \theta_t)
$$

在这种解耦方式中，Weight Decay 作用于权重更新的步骤，而不是通过损失函数的惩罚项来执行。这种方式被证明在一些深度学习任务中能带来更好的性能。

#### 学习率调度与 Weight Decay 联合使用

有时 Weight Decay 会与学习率调度器结合使用，形成动态衰减。学习率随时间减小的同时，Weight Decay 可以起到类似的作用，使模型收敛得更稳定。

---

#### 总结

Weight Decay 的常见形式包括：

- **L2 正则化**（最常见的形式）。
- **L1 正则化**（促使稀疏性）。
- **Elastic Net**（结合 L1 和 L2）。
- **Decoupled Weight Decay**（解耦更新）。
  此外，Dropout 等其他正则化方法也可以与这些权重衰减技术结合使用，以提高模型的泛化能力。



## Learning Rate

Learning Rate 是用于调整模型权重更新的步长，它决定了模型在每次迭代中权重调整的幅度。太大的学习率可能导致模型错过最优解，太小的学习率则会导致训练时间过长或陷入局部最优。

## Weight Decay 与 Learning Rate 的关系

Weight Decay 不是 Learning Rate 的一部分，但它们在权重更新过程中紧密相关。在使用 Weight Decay 的情况下，权重更新不仅依赖于梯度，还会包含一个衰减项。具体来说，梯度下降的更新公式变为：

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\partial L(\theta)}{\partial \theta} - \eta \cdot \lambda \theta_t
$$






其中：

- $\theta_{t+1}$ 是更新后的权重。
- $\theta_t$ 是当前的权重。
- $\eta$ 是学习率。
- $\lambda$ 是权重衰减系数。

这里，Weight Decay 通过 $- \eta \cdot \lambda \theta_t $部分来对权重进行额外的缩减，而学习率 $\eta$ 决定了衰减的速度。

- **Weight Decay** 是一种正则化手段，帮助防止过拟合，目的是控制权重的增长。
- **Learning Rate** 是控制模型学习速度的超参数，决定每次更新的步长。
- **它们在训练过程中是相互独立的超参数，但共同作用于模型的权重更新过程。**

因此，Weight Decay 并不属于学习率的一部分，但两者会共同影响模型的训练效果，通常需要同时调节它们以获得更好的模型表现。



> 在标准的随机梯度下降中，**权重衰减正则化和正则化**的效果相同
> 
> - 因此，权重衰减在一些深度学习框架中通过 L2 正则化来实现
> - 但是，在较为复杂的[优化方法](https://zhida.zhihu.com/search?q=%E4%BC%98%E5%8C%96%E6%96%B9%E6%B3%95&zhida_source=entity&is_preview=1)( 比如Adam ) 中，权重衰减正则化和正则化并不等价 [Loshchilov et al, 2017b]