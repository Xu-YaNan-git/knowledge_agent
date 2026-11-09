import torch

# 当前电脑没有检测到 NVIDIA 显卡，因此使用 CPU
device = torch.device("cpu")

a = torch.tensor([1.0, 2.0, 3.0], device=device)
b = torch.tensor([4.0, 5.0, 6.0], device=device)

print("PyTorch 版本:", torch.__version__)
print("相加:", a + b)
print("矩阵计算:", torch.matmul(a.unsqueeze(0), b.unsqueeze(1)))