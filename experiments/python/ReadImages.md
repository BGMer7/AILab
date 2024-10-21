# read images

在 PyTorch 中，读取和加载图片的方式有多种，通常使用 `torchvision` 提供的工具来加载和预处理图像数据。以下是一些常用的方法来读取和加载图片：

### 1. **`ImageFolder`**
   `torchvision.datasets.ImageFolder` 是一种常见的方式，它自动读取指定文件夹下按类别存放的图片，并将图片和标签对应起来。这种方法适用于已按类别分类的文件夹结构。

   #### 示例：
   ```python
   from torchvision import datasets, transforms
   from torch.utils.data import DataLoader

   # 定义图像的预处理操作
   transform = transforms.Compose([
       transforms.Resize((32, 32)),  # 调整图像大小
       transforms.ToTensor()  # 将图像转换为 Tensor
   ])

   # 使用 ImageFolder 读取图像
   dataset = datasets.ImageFolder(root='path/to/your/image_folder', transform=transform)

   # 创建 DataLoader 来批量加载图片
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

   # 读取一个批次的图像
   images, labels = next(iter(dataloader))
   print(images.shape)  # 输出形状: torch.Size([32, 3, 32, 32])
   ```

### 2. **`PIL.Image`**
   `PIL`（Python Imaging Library）可以直接读取单个图片文件。PyTorch 中，`torchvision.transforms` 中的 `ToTensor()` 可以将 `PIL` 图像转换为 PyTorch 的张量。

   #### 示例：
   ```python
   from PIL import Image
   import torchvision.transforms as transforms

   # 打开图像
   img = Image.open('path/to/your/image.jpg')

   # 转换为 Tensor
   transform = transforms.ToTensor()
   img_tensor = transform(img)
   print(img_tensor.shape)  # 输出图像形状：torch.Size([3, H, W])
   ```

### 3. **`cv2.imread()` (OpenCV)**
   使用 OpenCV 也可以读取图片，并且可以轻松进行各种图像处理操作。需要注意的是，OpenCV 读取的图片通道顺序是 `BGR`，而 PyTorch 需要的是 `RGB`。

   #### 示例：
   ```python
   import cv2
   import torchvision.transforms as transforms
   import torch

   # 使用 OpenCV 读取图片
   img = cv2.imread('path/to/your/image.jpg')

   # 将 BGR 转换为 RGB
   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

   # 转换为 Tensor
   transform = transforms.ToTensor()
   img_tensor = transform(img_rgb)
   print(img_tensor.shape)  # 输出形状: torch.Size([3, H, W])
   ```

### 4. **`matplotlib.pyplot.imread()`**
   `matplotlib` 提供了读取图像的简单方法 `imread()`，它可以读取图片并返回 NumPy 数组。

   #### 示例：
   ```python
   import matplotlib.pyplot as plt
   import torchvision.transforms as transforms

   # 使用 matplotlib 读取图片
   img = plt.imread('path/to/your/image.jpg')

   # 将 NumPy 数组转换为 PyTorch Tensor
   transform = transforms.ToTensor()
   img_tensor = transform(img)
   print(img_tensor.shape)  # 输出形状: torch.Size([3, H, W])
   ```

### 5. **自定义图片读取器**
   有时，您可能需要自定义加载图像的方法，特别是在数据集没有现成的文件夹结构时。可以通过读取文件路径列表，自己定义图像加载的逻辑。

   #### 示例：
   ```python
   from PIL import Image
   import os
   import torchvision.transforms as transforms

   # 自定义函数读取图片
   def load_image(image_path):
       img = Image.open(image_path)
       return img

   # 读取图像列表
   image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
   images = [load_image(path) for path in image_paths]

   # 转换为 Tensor
   transform = transforms.ToTensor()
   images_tensor = [transform(img) for img in images]

   # 显示第一张图片的形状
   print(images_tensor[0].shape)  # torch.Size([3, H, W])
   ```

### 6. **`skimage.io.imread()`**
   `skimage` 是另一个流行的图像处理库，也可以用来读取图像并返回 NumPy 数组。

   #### 示例：
   ```python
   from skimage import io
   import torchvision.transforms as transforms

   # 使用 skimage 读取图片
   img = io.imread('path/to/your/image.jpg')

   # 将 NumPy 数组转换为 Tensor
   transform = transforms.ToTensor()
   img_tensor = transform(img)
   print(img_tensor.shape)  # 输出形状: torch.Size([3, H, W])
   ```

### 总结
- **`ImageFolder`** 是 PyTorch 提供的方便方式，可以批量处理按类别分类的图像。
- **`PIL.Image`**、**`cv2.imread()`**、**`matplotlib.pyplot.imread()`** 和 **`skimage.io.imread()`** 都是读取单张图片的常用方法。
- 可以根据具体需求选择合适的工具进行图像读取与处理，最后通过 `torchvision.transforms` 将图片转换为 PyTorch 的张量，以便后续在模型中使用。





# show images

`matplotlib.pyplot` 支持展示多种类型的图片，常见的图片类型包括：

### 1. **NumPy 数组 (`numpy.ndarray`)**
   `matplotlib.pyplot.imshow()` 最常用的是展示以 NumPy 数组形式表示的图片，特别是通过读取图像文件或者图像处理后的数据。

   #### 示例：
   ```python
   import matplotlib.pyplot as plt
   import numpy as np

   # 创建一个随机的 NumPy 数组表示图像
   img = np.random.rand(100, 100, 3)  # 形状为 (H, W, 3) 的 RGB 图片

   # 使用 imshow 显示图像
   plt.imshow(img)
   plt.show()
   ```

### 2. **PIL 图像对象**
   如果使用 `PIL.Image` 打开图像，`matplotlib` 也能直接显示 `PIL` 图像对象。

   #### 示例：
   ```python
   from PIL import Image
   import matplotlib.pyplot as plt

   # 打开图像
   img = Image.open('path/to/your/image.jpg')

   # 显示图像
   plt.imshow(img)
   plt.axis('off')  # 隐藏坐标轴
   plt.show()
   ```

### 3. **PyTorch Tensor**
   PyTorch 的张量（`torch.Tensor`）在通过 `.permute()` 调整维度顺序后，可以通过 `matplotlib` 显示。这是因为 `PyTorch` 张量的维度顺序通常是 `[C, H, W]`（通道数、图像高度、图像宽度），而 `matplotlib` 期望的是 `[H, W, C]`（图像高度、图像宽度、通道数）。

   #### 示例：
   ```python
   import torch
   import matplotlib.pyplot as plt

   # 创建一个随机的 PyTorch 张量表示图像
   img_tensor = torch.rand(3, 100, 100)  # 形状为 [C, H, W]

   # 调整张量维度顺序以适应 matplotlib 的格式
   img_tensor = img_tensor.permute(1, 2, 0)  # 形状变为 [H, W, C]

   # 显示图像
   plt.imshow(img_tensor)
   plt.show()
   ```

### 4. **灰度图像**
   无论是 NumPy 数组、PIL 图像还是 PyTorch 张量，灰度图像也可以直接展示。灰度图像通常只有两个维度 `[H, W]`，而不是 RGB 图片的三个维度 `[H, W, C]`。

   #### 示例：
   ```python
   import numpy as np
   import matplotlib.pyplot as plt

   # 创建一个灰度图像的 NumPy 数组
   gray_img = np.random.rand(100, 100)  # 形状为 (H, W)

   # 显示灰度图像
   plt.imshow(gray_img, cmap='gray')  # 使用灰度颜色映射
   plt.show()
   ```

### 5. **二值图像 (Binary Images)**
   `matplotlib` 也支持展示二值图像，二值图像中的像素值通常是 0 或 1，表示黑白图像。

   #### 示例：
   ```python
   import numpy as np
   import matplotlib.pyplot as plt

   # 创建一个二值图像的 NumPy 数组
   binary_img = np.random.randint(0, 2, (100, 100))  # 形状为 (H, W)，值为 0 或 1

   # 显示二值图像
   plt.imshow(binary_img, cmap='gray')  # 使用灰度颜色映射
   plt.show()
   ```

### 6. **多个图像（例如通过 `make_grid` 合并的图像）**
   如果通过 `torchvision.utils.make_grid()` 将多个图像合并为一个网格，`matplotlib` 也可以显示此网格形式的图片。

   #### 示例：
   ```python
   from torchvision.utils import make_grid
   import matplotlib.pyplot as plt
   import torch

   # 创建一个 4D 的随机张量 (batch_size, C, H, W)
   images = torch.rand(16, 3, 32, 32)

   # 使用 make_grid 生成图像网格
   img_grid = make_grid(images, nrow=4)  # 4 行图像

   # 调整维度并显示
   plt.imshow(img_grid.permute(1, 2, 0))
   plt.show()
   ```

### 7. **其他格式（如 PNG、JPEG 等）**
   `matplotlib` 通过 `imshow()` 也支持展示常见图像格式（如 PNG、JPEG），通常在使用 `PIL.Image` 进行加载后展示。

### 总结
- `matplotlib.pyplot.imshow()` 主要支持 NumPy 数组、PIL 图像对象以及 PyTorch 张量。
- 还可以处理灰度图像、二值图像，以及通过图像网格函数（如 `make_grid`）生成的多张图像拼接后的效果。
