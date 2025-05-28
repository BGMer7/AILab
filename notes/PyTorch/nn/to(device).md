当你调用一个 PyTorch `nn.Module` 对象的 `.to(device)` 方法（例如 `model.to('cuda')`）时，以下类型的张量会被转移到指定的设备（如 CUDA GPU）：

1. **通过 `register_parameter()` 注册的参数 (Parameters)**:
   
    - 这些是模型中需要学习的权重和偏置。
    - 它们通常是通过 `nn.Parameter(torch.Tensor(...))` 创建并赋值给模块的属性，或者通过 `self.register_parameter("param_name", nn.Parameter(torch.Tensor(...)))` 显式注册。
    - **示例**: `self.weight = nn.Parameter(torch.randn(10, 5))` 或 `self.register_parameter("bias", nn.Parameter(torch.zeros(5)))`。
2. **通过 `register_buffer()` 注册的缓冲区 (Buffers)**:
   
    - 这些是模型状态的一部分，但不需要梯度更新的张量，例如批量归一化层中的 `running_mean` 和 `running_var`。
    - 它们是通过 `self.register_buffer("buffer_name", torch.Tensor(...))` 注册的。
    - **示例**: `self.register_buffer("running_mean", torch.zeros(num_features))`。
3. **递归地应用于所有子模块 (submodules)**:
   - 如果你的模型包含其他 `nn.Module` 实例作为其属性（例如，一个包含多个卷积层和线性层的网络），`model.to(device)` 调用会递归地将这个方法应用于所有子模块。
    - 因此，子模块中的所有参数和缓冲区也会被移动到指定的设备。

**哪些不会被自动转移？**

- **普通的 Python 属性 (包括普通的 PyTorch 张量)**: 如果你仅仅将一个 PyTorch 张量赋值给模块的一个属性，而没有使用 `nn.Parameter()` 包装它，也没有使用 `register_parameter()` 或 `register_buffer()` 注册它，那么这个张量**不会**被 `model.to(device)` 自动转移。
  
  
    ```python
    import torch
    import torch.nn as nn
    
    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.my_param = nn.Parameter(torch.randn(2, 2)) # 会转移
            self.register_buffer("my_buffer", torch.randn(3, 3)) # 会转移
            self.regular_tensor = torch.randn(4, 4) # 不会通过 model.to() 自动转移
            self.some_list = [torch.randn(1), torch.randn(1)] # 列表中的张量不会自动转移
    
        def forward(self, x):
            # 在实际使用中，你需要确保所有参与计算的张量都在同一个设备上
            # 如果 regular_tensor 没有手动移到 GPU，而 x 和 my_param 在 GPU，这里会报错
            # return x @ self.my_param + self.my_buffer[0,0] * self.regular_tensor
            return x @ self.my_param + self.my_buffer[0,0] * self.regular_tensor.to(x.device) # 正确做法
    
    model = MyModule()
    print(f"Initial device of my_param: {model.my_param.device}")
    print(f"Initial device of my_buffer: {model.my_buffer.device}")
    print(f"Initial device of regular_tensor: {model.regular_tensor.device}")
    
    if torch.cuda.is_available():
        model.to('cuda')
        print(f"\nDevice of my_param after .to('cuda'): {model.my_param.device}")
        print(f"Device of my_buffer after .to('cuda'): {model.my_buffer.device}")
        print(f"Device of regular_tensor after .to('cuda'): {model.regular_tensor.device}") # 仍然是 cpu
        # 需要手动转移:
        # model.regular_tensor = model.regular_tensor.to('cuda')
        # print(f"Device of regular_tensor after manual .to('cuda'): {model.regular_tensor.device}")
    else:
        print("\nCUDA not available for demonstration.")
    ```
    
- **其他 Python 对象**: 非张量类型的 Python 对象（如列表、字典、整数、字符串等）自然不会被 `.to(device)` 方法影响，因为它们不驻留在特定的计算设备上。如果这些对象包含了张量（例如一个包含张量的列表），列表本身不会被转移，你需要手动遍历并转移列表中的每个张量。

**总结一下：**

当你调用 `model.to(device)` 时，PyTorch 会自动将所有被正确注册为模型一部分的**参数 (Parameters)** 和**缓冲区 (Buffers)**（以及所有子模块的参数和缓冲区）移动到目标设备。对于未注册的普通张量属性，你需要手动调用 `.to(device)` 进行转移。