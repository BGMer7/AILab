《动手学深度学习v2》 pytorch版

https://zh.d2l.ai/index.html

pytorch和cuda

https://www.bilibili.com/read/cv34265365/?spm_id_from=..0.0

## torch gpu install 
在安装GPU版本的torch过程中，d2l和torch还有torchvision发生了一系列的冲突，导致各处的代码存在报错。

主要冲突的库包括torch、torchvision、numpy和d2l。

最终遵循以下安装原则，基本做到兼容：

1. 安装pytorhc
    
    根据显卡型号、OS版本、cuda硬件型号，cuda软件型号，确定pytorch的安装命令，这一系列操作可以在pytorch官网选择，并最终生成命令直接复制执行。
      ```shell
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
      ```
2. 安装d2l
   
    后来是在纯净环境里安装的torch环境，安装完成之后再使用pip安装的d2l，一来是conda没有d2l的库，二来是pip不容易覆盖别的库的版本。
   ```shell
   pip install -U d2l
    ```