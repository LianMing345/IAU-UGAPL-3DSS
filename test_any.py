import torch

a = torch.Tensor([1, 2, 1])
b = torch.Tensor([2, 1, 3])
print(torch.max(a, b))