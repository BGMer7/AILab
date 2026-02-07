# 首先需要安装 modelscope： pip install modelscope
from modelscope import snapshot_download

# 下载 Qwen2.5-7B-Instruct 版本
# 注意：做意图识别建议下载 Instruct 版本，因为它已经懂对话了，微调收敛更快
model_dir = snapshot_download(
    "qwen/Qwen3-0.6B",
    cache_dir="./models",  # 下载到当前目录下的 models 文件夹
    revision="master",  # 下载最新版本
)

print(f"模型已下载到: {model_dir}")
