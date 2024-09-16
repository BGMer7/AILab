【比啃书效果好多了！复旦大学邱锡鹏教授的《神经网络与深度学习》终于出视频教程了！！草履虫都能看懂！】 https://www.bilibili.com/video/BV1P3411u7c1/?p=36&share_source=copy_web&vd_source=975386ba054f2685c1a99769524e2403

# Convolution

## Background

卷积在许多不同的领域和场景中有广泛的应用，特别是在信号处理、图像处理和深度学习中。它经常用于设计和应用**滤波器**，其目的是对输入信号进行处理，以提取重要特征或去除不需要的噪声。下面是一些卷积的典型应用背景：

### 信号处理中的滤波器

卷积在信号处理中的一个主要用途是设计和应用滤波器。滤波器的作用是对信号进行操作，如消除噪声、提取特定频率或平滑信号。卷积在这类滤波操作中起着核心作用。

- **低通滤波器（Low-Pass Filter, LPF）**：允许低频信号通过，滤除高频成分，从而平滑信号或消除噪声。
- **高通滤波器（High-Pass Filter, HPF）**：允许高频信号通过，滤除低频成分，通常用于边缘检测或提高信号中的细节。
- **带通滤波器（Band-Pass Filter）**：只允许一定频率范围内的信号通过，滤除超出该范围的其他频率。

举例：

- 在音频处理领域，卷积滤波器可以用来减少录音中的噪声或增强某些音频频率。
- 在通信系统中，滤波器通过卷积消除通信信号中的干扰或不需要的频率成分。

### 图像处理

卷积在图像处理中扮演了至关重要的角色，尤其在**边缘检测**、**模糊处理**、**锐化**等方面。通过应用卷积滤波器，可以提取出图像中的关键特征，或对图像进行某种增强或抑制。

- **边缘检测**：通过高通滤波器（如 Sobel、Laplacian 卷积核），可以识别图像中亮度急剧变化的区域，这些区域通常是图像的边缘。边缘检测广泛用于图像识别、目标检测等任务。
  
  - **Sobel算子**（用于检测水平和垂直边缘）：
    
    - 水平边缘检测核
    
    $$
    \begin{bmatrix}
    -1 & 0 & 1 \\
    -2 & 0 & 2 \\
    -1 & 0 & 1
    \end{bmatrix}
    $$
    
    - 垂直边缘检测核
    
    $$
    \begin{bmatrix}
    -1 & -2 & -1 \\
    0 & 0 & 0 \\
    1 & 2 & 1
    \end{bmatrix}
    $$

- **模糊滤波器（Blurring Filter）**：卷积核可以用于图像模糊处理，常用于减少图像中的噪声或平滑图像，模糊滤波器通常是低通滤波器。
  
  - **均值滤波器（Mean Filter）**：
    
    $$
    \frac{1}{9}
    \begin{bmatrix}
    1 & 1 & 1 \\
    1 & 1 & 1 \\
    1 & 1 & 1
    \end{bmatrix}
    $$
    
    这种滤波器会将图像中的每个像素与其邻域平均化，从而降低图像的锐度和细节。

- **锐化滤波器（Sharpening Filter）**：通过增强图像的边缘信息，使得图像中的细节更加清晰。
  
  - **锐化卷积核**：
    
    $$
    \begin{bmatrix}
    0 & -1 & 0 \\
    -1 & 5 & -1 \\
    0 & -1 & 0
    \end{bmatrix}
    $$

举例：

- 在医学成像中，边缘检测有助于识别病变组织或器官的轮廓。
- 在计算机视觉中，模糊滤波用于减少噪声、提高后续处理（如对象识别或分类）的准确性。

### CNN

在机器学习特别是深度学习领域，卷积神经网络（CNN）广泛用于**图像分类**、**目标检测**、**语音识别**等任务。CNN 的核心是通过卷积层对输入图像进行特征提取，每一层卷积都从输入中提取不同层次的特征。

- **第一层卷积**：通常提取低级别特征，如边缘、线条等。
- **中间层卷积**：提取更复杂的特征，如纹理、形状等。
- **深层卷积**：提取高阶语义特征，如物体的整体轮廓或类别。

CNN 中的卷积滤波器权重是通过训练数据学习而来的，不同于传统的手动设计滤波器。每个卷积层会生成一个**特征图**（Feature Map），这些特征图捕捉了输入图像的不同方面。

举例：

- 在图像分类中，CNN 可以通过多个卷积层逐步提取图像中的高层次特征，并最终通过全连接层输出图像的类别。
- 在自动驾驶中，CNN 能够从摄像头获取的图像中检测出路标、行人等重要信息。

### 音频处理

卷积在音频处理和合成中也有广泛的应用，特别是用于设计滤波器和特效。

- **卷积混响（Convolution Reverb）**：将一个录制的环境（如教堂、音乐厅等）的声学特性与输入音频信号进行卷积，可以模拟出该环境下的声音效果。

- **频率滤波**：通过卷积滤波器可以提取或抑制某些特定频率的音频信号，这在音频修复、降噪、语音识别等领域十分重要。

举例：

- 在音乐制作中，卷积混响可以使录制的声音具有不同环境的混响效果，从而产生丰富的音质。
- 在语音识别系统中，卷积滤波器可以帮助提取语音中的关键特征，并减少背景噪音的干扰。

### 其他领域

- **时间序列分析**：在经济学、金融市场等领域，卷积可以用于对时间序列数据进行平滑、降噪或提取趋势。
- **雷达信号处理**：卷积用于从复杂的雷达回波信号中提取目标物的信息，如位置、速度等。
- **生物信息学**：在基因组序列的分析中，卷积可以用于寻找特定的模式或特征，如基因序列中的重复序列或特定结构。

---

**总结**

卷积在各种背景下，尤其是滤波器的应用中，是一种强大的数学工具。无论是在图像、音频、信号处理，还是深度学习中，卷积通过局部操作提取数据中的重要特征，并可以根据具体任务设计不同的卷积核或滤波器，以达到去噪、增强、特征提取等目的。

## Operation

卷积操作是深度学习（尤其是卷积神经网络，CNN）中的核心步骤，广泛应用于图像处理、音频信号处理等领域。卷积操作可以用来提取输入信号（如图像、音频）的局部特征。

### Process

#### Kernel

- 卷积核（Convolutional Kernel 或 Filter）是一个小矩阵，通常大小为 3x3 或 5x5，由可学习的权重组成。
- 卷积操作的目的是通过卷积核与输入进行局部点乘运算，得到输出的特征图。

#### Stride

- 步幅决定了卷积核每次滑动的距离。
  - **步幅为 1**：卷积核每次移动一个像素。
  - **步幅为 2**：卷积核每次移动两个像素。
- 步幅越大，输出特征图的尺寸越小。

#### Padding

- 填充是在输入矩阵的边缘填充额外的像素（**通常为 0**），以保持输出的尺寸不变或减少过多信息丢失。
  - **SAME 填充**：填充使得输入输出尺寸相同。
  - **VALID 填充**：不填充，卷积核只在输入的有效区域滑动。

#### dot product

卷积是将卷积核在输入图像上滑动，并对每个位置进行局部的点积运算：

- 卷积核在输入图像上每移动一次，核内的数值和输入图像对应区域的数值相乘并累加，得到一个数值，放入输出的特征图中。
- 通过重复这个操作，卷积核遍历整个输入图像，生成输出特征图。

### Sample

假设我们有一个 5x5 的输入矩阵和一个 3x3 的卷积核，步幅为 1，填充为 0（VALID padding），卷积过程如下：

**输入矩阵：**
$$
\begin{bmatrix}
1 & 2 & 3 & 0 & 1 \\
0 & 1 & 2 & 3 & 1 \\
1 & 2 & 1 & 0 & 2 \\
0 & 1 & 2 & 1 & 1 \\
2 & 1 & 0 & 2 & 1
\end{bmatrix}
$$
**卷积核：**
$$
\begin{bmatrix}
0 & 1 & 2 \\
1 & 0 & 1 \\
2 & 1 & 0
\end{bmatrix}
$$

#### Calculation

1. 将卷积核放在输入矩阵的左上角，计算其覆盖区域内的点乘：
   
   $$
   (1\cdot0) + (2\cdot1) + (3\cdot2) + (0\cdot1) + (1\cdot0) + (2\cdot1) + (1\cdot2) + (2\cdot1) + (1\cdot0) = 0 + 2 + 6 + 0 + 0 + 2 + 2 + 2 + 0 = 14
   $$
   
   将结果 14 放入输出矩阵的相应位置。

2. 卷积核向右移动一个步幅，重复相同操作，直到卷积核遍历完整个输入矩阵。

> 实际上，真正的用于滤波的卷积都是翻转计算的。
>
> 因为最初卷积滤波器的作用是计算信号的衰减，详见邱锡鹏老师的课程内容【比啃书效果好多了！复旦大学邱锡鹏教授的《神经网络与深度学习》终于出视频教程了！！草履虫都能看懂！】 https://www.bilibili.com/video/BV1P3411u7c1/?p=36&share_source=copy_web&vd_source=975386ba054f2685c1a99769524e2403
>
> 由于信号不停衰减，所以t时间之后的信号需要加上一个权重，这个权重随着t的推移是越来越小的，导致最新的信号乘上最大的信号权重，最老的信号乘上最小的信号权重。
>
> 在计算机中的卷积一般使用的是互相关计算，互相关计算可以近似理解为不停进行矩阵计算，是无需翻转的。



#### Output

$$
\begin{bmatrix}
14 & 13 & 11 \\
10 & 14 & 16 \\
9  & 10 & 12
\end{bmatrix}
$$

## Concepts

### Local Connectivity

局部连接

每个输出神经元只连接输入的一小部分，减少参数和计算量。



#### Respective Field

- 每个卷积核只在输入数据的一个小区域上滑动，并与该区域的值进行点乘。这个小区域称为**感受野**。
- 感受野的大小由卷积核的尺寸（例如 3x3 或 5x5）决定。每个输出神经元只感知输入数据的一个局部区域，而不是全局区域。
- 感受野也是局部连接思想的一部分。

#### Weight Sharing

权重共享

- 卷积核（或滤波器）通过在输入上滑动来覆盖不同的局部区域。在每一个局部区域，卷积核应用相同的权重进行计算。这意味着在所有位置使用相同的卷积核，称为**权重共享**

同一个卷积核在整个输入上滑动，提取共享的特征，这大大减少了模型的参数数量。



- 由于卷积层中的每个神经元只连接到局部区域，CNN 不需要为每个输入像素都存储独立的权重，而是通过卷积核在不同区域滑动共享权重。这大大减少了网络中的参数数量。
- 比如，一个 5x5 的卷积核只有 25 个参数，而它可以通过滑动覆盖整个图像，从而极大减少计算量。



### 平移不变性

卷积操作能够捕捉到相同的特征在不同位置的存在，从而具有平移不变性。





### 一维卷积

$$
y(t) = (x * h)(t) = \sum_{\tau=-\infty}^{\infty} x(\tau) h(t - \tau)
$$

### 二维卷积

$$
y(i, j) = (x * h)(i, j) = \sum_m \sum_n x(i - m, j - n) h(m, n)
$$



### Feature Map

在卷积神经网络（CNN）中，**Feature Map**（特征图）是每一层卷积操作后的输出，是从输入数据中提取出来的特征。Feature Map 是 CNN 中非常关键的概念，主要反映了卷积层如何捕捉图像或其他输入数据的局部特征，如边缘、纹理、形状等。

#### Feature Map 的生成

Feature Map 是通过卷积核（滤波器）与输入数据进行卷积计算得到的，具体步骤如下：

1. **输入图像**：假设输入数据是一张图像，它的每个像素可以被看作是一个数值。
2. **卷积核（Filter/Kernel）**：卷积核是一组权重参数的矩阵，大小通常为 3x3、5x5 等。每个卷积核用于识别图像的某种特定模式，比如垂直边缘、水平边缘等。
3. **卷积操作**：卷积核会在输入图像上滑动，通过在局部区域内对图像的像素值和卷积核进行点积计算，生成输出值。这些输出值构成了特征图的每个位置的值。
4. **特征提取**：滑动卷积核遍历整个图像后，得到一个新的矩阵，这个矩阵称为**Feature Map**，表示输入图像在该卷积核下提取出的特征。

#### Feature Map 的特点

1. **局部感知**：
   - 每个卷积核在图像的局部区域上滑动并提取特征，这使得每个 Feature Map 反映了局部区域内的信息，如边缘、角点等。
   
2. **深度增加**：
   - 不同的卷积核可以提取不同类型的特征。CNN 中通常有多个卷积核（即多个滤波器），每个卷积核产生一个特征图。经过多个卷积核后，图像的特征可以被更加深层次地提取。
   
3. **空间维度减小**：
   - 特征图的空间维度（宽度和高度）通常会比原始输入图像小，因为卷积操作会减少图像的边缘信息。此外，池化操作（Pooling）也会进一步减小特征图的大小。

4. **多通道**：
   - 如果输入图像是彩色的（RGB 图像），则输入数据有多个通道（3个通道分别表示红色、绿色和蓝色）。卷积操作会分别对每个通道应用卷积核，生成对应的特征图。

#### Feature Map 的可视化

为了更好地理解 CNN 学到的特征，研究人员通常会对不同卷积层输出的 Feature Map 进行可视化：

- **浅层特征**：靠近输入层的卷积层往往学习到一些低级特征（例如，边缘、纹理等）。
- **深层特征**：靠近输出层的卷积层则学习到更复杂的高级特征，如图像中的形状、对象等。

#### Feature Map 的代码示例

以下是一个简单的例子，演示如何使用卷积神经网络提取特征图：

```python
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt

# 使用预训练的ResNet18模型
model = models.resnet18(pretrained=True)

# 打印模型结构
print(model)

# 选取前几层作为卷积部分
conv_layers = list(model.children())[:4]
model = nn.Sequential(*conv_layers)

# 输入一个示例图像
input_image = torch.randn(1, 3, 224, 224)  # 假设输入是一个3通道的224x224图像

# 前向传播，生成特征图
feature_map = model(input_image)

# 可视化特征图
# 我们选择第一个卷积核的输出特征图
plt.imshow(feature_map[0, 0].detach().numpy(), cmap='gray')
plt.show()
```

在这个示例中，我们使用预训练的 ResNet18 模型，并通过提取前几层的卷积操作来生成特征图。



## Kernel

**Sobel 算子** 和 **Prewitt 算子** 是常见的用于边缘检测的算子，它们通过卷积操作检测图像中局部的梯度变化，从而识别边缘。两者的主要区别在于计算梯度时使用的权重略有不同，但核心思想相同。下面是对这两个算子的详细解释。

### Sobel 算子

**Sobel 算子** 是一个离散的差分算子，通常用于计算图像的**梯度**，以增强边缘的效果。它通过在水平方向和垂直方向分别应用不同的卷积核，来检测图像的水平和垂直边缘。

- **水平方向卷积核 (Sobel-X)**：
  
  $$
  \begin{bmatrix}
  -1 & 0 & 1 \\
  -2 & 0 & 2 \\
  -1 & 0 & 1
  \end{bmatrix}
  $$

- **垂直方向卷积核 (Sobel-Y)**：
  $$
  \begin{bmatrix}
  -1 & -2 & -1 \\
   0 &  0 &  0 \\
   1 &  2 &  1
  \end{bmatrix}
  $$

#### Sobel 算子的特点：
- 中心权重为 0，表示不考虑中心点。
- 边缘部分的权重为 2，使得对图像边缘的响应更加明显。
- **增强边缘效果**：相比于 Prewitt 算子，Sobel 算子对噪声的抵抗能力稍强，因为它更强调靠近中心的像素。

### Prewitt 算子

**Prewitt 算子** 是另一种边缘检测算子，与 Sobel 算子非常相似，主要的区别在于它对不同方向的梯度计算使用的卷积核不同。Prewitt 算子通过计算水平方向和垂直方向的梯度来检测边缘。

- **水平方向卷积核 (Prewitt-X)**：
  $$
  \begin{bmatrix}
  -1 & 0 & 1 \\
  -1 & 0 & 1 \\
  -1 & 0 & 1
  \end{bmatrix}
  $$
  
- **垂直方向卷积核 (Prewitt-Y)**：
  
  $$
  \begin{bmatrix}
  -1 & -1 & -1 \\
   0 &  0 &  0 \\
   1 &  1 &  1
  \end{bmatrix}
  $$

#### Prewitt 算子的特点：
- 中心权重为 0，与 Sobel 算子类似，但边缘权重都为 1。
- **更加简单**：Prewitt 算子没有像 Sobel 算子那样对边缘像素进行加权处理，因此计算速度更快，但对噪声的抵抗力稍差。

### 梯度计算

无论是 Sobel 还是 Prewitt 算子，它们的梯度计算方式都是一样的。给定水平方向梯度 `Gx` 和垂直方向梯度 `Gy`，边缘强度可以通过以下公式计算：

$$
G = \sqrt{G_x^2 + G_y^2}
$$
边缘的方向角度 `θ` 可以通过下式计算：

$$
\theta = \text{atan2}(G_y, G_x)
$$
其中，`Gx` 是使用水平方向的卷积核，`Gy` 是使用垂直方向的卷积核。

### Sobel 和 Prewitt 的比较

| 特性       | Sobel 算子                     | Prewitt 算子                |
| ---------- | ------------------------------ | --------------------------- |
| 权重分布   | 中心较大，边缘为 2，抗噪能力强 | 权重均匀分布，较简单        |
| 精确度     | 对边缘的检测更精确             | 检测边缘不如 Sobel 算子精确 |
| 抗噪性     | 更强的抗噪能力                 | 抗噪能力较弱                |
| 计算复杂度 | 略高于 Prewitt 算子            | 计算较简单，速度较快        |
| 应用场景   | 用于更复杂的图像处理           | 适用于简单图像处理          |

### 使用示例

以下是使用 PyTorch 实现 Sobel 算子来检测图像边缘的代码示例：

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 定义 Sobel 算子的卷积核
sobel_x = torch.tensor([[-1., 0., 1.], 
                        [-2., 0., 2.], 
                        [-1., 0., 1.]]).view(1, 1, 3, 3)

sobel_y = torch.tensor([[-1., -2., -1.], 
                        [ 0.,  0.,  0.], 
                        [ 1.,  2.,  1.]]).view(1, 1, 3, 3)

# 输入图像 (随机生成的3x3灰度图像)
input_image = torch.randn(1, 1, 5, 5)

# 使用 Sobel 卷积核分别检测水平和垂直边缘
output_x = F.conv2d(input_image, sobel_x)
output_y = F.conv2d(input_image, sobel_y)

# 计算梯度强度
edge_strength = torch.sqrt(output_x ** 2 + output_y ** 2)

# 可视化结果
plt.subplot(1, 3, 1)
plt.title('Input Image')
plt.imshow(input_image[0][0].detach().numpy(), cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Edge Detection (Sobel-X)')
plt.imshow(output_x[0][0].detach().numpy(), cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Edge Detection (Sobel-Y)')
plt.imshow(output_y[0][0].detach().numpy(), cmap='gray')

plt.show()
```





## Datasets

在深度学习库中，确实有一些内置的常用数据集可以直接导入用于测试模型。以下是一些流行的深度学习框架以及它们提供的常见数据集：

### TensorFlow / Keras
TensorFlow 和 Keras 都内置了一些常见的数据集，特别是用于图像分类的经典数据集。

- **导入内置数据集示例**：
  TensorFlow 提供了多种内置数据集，可以通过 `tf.keras.datasets` 直接导入。

  ```python
  import tensorflow as tf
  
  # 加载 MNIST 数据集
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
  
  # 加载 CIFAR-10 数据集
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
  
  # 加载 Fashion MNIST 数据集
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
  ```

  ### 常见内置数据集：
  - **MNIST**: 手写数字分类数据集（28x28 灰度图像，10 个类别）。
  - **CIFAR-10**: 10 个类别的小型彩色图片数据集（32x32 RGB 图像）。
  - **CIFAR-100**: 类似于 CIFAR-10，但包含 100 个类别。
  - **Fashion MNIST**: 类似于 MNIST，但包含服装类别的图片数据集。
  - **IMDB**: 用于情感分析的电影评论文本数据集。
  - **Boston Housing**: 房价回归任务数据集。

### PyTorch
PyTorch 也内置了多个数据集，通过 `torchvision.datasets` 可以轻松导入。

- **导入内置数据集示例**：
  ```python
  import torch
  from torchvision import datasets, transforms
  
  # 定义图像的预处理方式
  transform = transforms.Compose([transforms.ToTensor()])
  
  # 加载 MNIST 数据集
  train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
  test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
  
  # 加载 CIFAR-10 数据集
  train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
  test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
  ```

  ### 常见内置数据集：
  - **MNIST**: 手写数字数据集。
  - **CIFAR-10** 和 **CIFAR-100**: 10 类或 100 类彩色图片。
  - **Fashion MNIST**: 用于服装图像分类。
  - **ImageNet**（需要手动下载）: 大规模的图片分类数据集。
  - **COCO**: 大规模目标检测、分割等任务的数据集。
  - **VOC**: Pascal VOC，图像分类和目标检测数据集。

### Fastai
Fastai 也是基于 PyTorch 的高层库，它提供了更简便的方式来加载和使用内置数据集。

- **导入内置数据集示例**：
  ```python
  from fastai.vision.all import *
  
  # 从 Fastai 提供的数据集中加载 MNIST
  path = untar_data(URLs.MNIST_SAMPLE)
  data = ImageDataLoaders.from_folder(path)
  
  # 加载 CIFAR-10 数据集
  path = untar_data(URLs.CIFAR)
  data = ImageDataLoaders.from_folder(path)
  ```

  ### 常见内置数据集：
  - **MNIST_SAMPLE**: 一个小规模的 MNIST 子集。
  - **CIFAR-10**: 彩色图像数据集。
  - **ImageWoof**: ImageNet 的一个子集，包含狗的类别。
  - **ImageNet**（需要手动下载）: 大型分类数据集。

### Hugging Face Datasets
`Hugging Face` 提供了一个名为 `datasets` 的库，集成了大量机器学习数据集，涵盖文本、图像、音频等多种领域。

- **导入内置数据集示例**：
  ```python
  from datasets import load_dataset
  
  # 加载一个文本数据集 (IMDB)
  dataset = load_dataset("imdb")
  
  # 加载一个图像数据集 (CIFAR-10)
  dataset = load_dataset("cifar10")
  ```

  ### 常见内置数据集：
  - **IMDB**: 文本情感分类数据集。
  - **CIFAR-10**: 图像分类数据集。
  - **SQuAD**: 用于问答系统的文本数据集。
  - **Common Voice**: 语音数据集，用于语音识别。

### TensorFlow Datasets (TFDS)
`TensorFlow Datasets` 是一个用于导入各种数据集的 TensorFlow 库，支持超过 100 种数据集，涵盖了文本、图像、视频等。

- **导入内置数据集示例**：
  ```python
  import tensorflow_datasets as tfds
  
  # 加载 MNIST 数据集
  dataset = tfds.load('mnist', split='train', as_supervised=True)
  
  # 加载 CIFAR-10 数据集
  dataset = tfds.load('cifar10', split='train', as_supervised=True)
  ```

  ### 常见内置数据集：
  - **MNIST**: 手写数字数据集。
  - **CIFAR-10** 和 **CIFAR-100**: 彩色图像数据集。
  - **CelebA**: 包含名人脸部图像的数据集。
  - **SVHN**: 街景门牌号数据集。
  - **Oxford Flowers 102**: 包含 102 种花类的图像分类数据集。

### 总结
以下是深度学习库中常见内置数据集的汇总：

| 框架                           | 常见内置数据集                                               |
| ------------------------------ | ------------------------------------------------------------ |
| **TensorFlow/Keras**           | MNIST, CIFAR-10, CIFAR-100, Fashion MNIST, IMDB, Boston Housing |
| **PyTorch**                    | MNIST, CIFAR-10, CIFAR-100, Fashion MNIST, ImageNet (手动下载), VOC, COCO |
| **Fastai**                     | MNIST_SAMPLE, CIFAR-10, ImageWoof, ImageNet (手动下载)       |
| **Hugging Face**               | IMDB, CIFAR-10, SQuAD, Common Voice                          |
| **TensorFlow Datasets (TFDS)** | MNIST, CIFAR-10, CelebA, SVHN, Oxford Flowers 102            |

这些数据集可以帮助快速测试和验证模型。根据需求选择合适的框架和数据集，便于实验和模型评估。
