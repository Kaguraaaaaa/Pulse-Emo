
import torch

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    print("CUDA is available! Running on GPU.")
    print("CUDA version:", torch.version.cuda)
    print("CUDA device count:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")