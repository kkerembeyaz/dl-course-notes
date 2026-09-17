import torch
import numpy as np

tensor = torch.tensor([[1, 2, 3],
                     [4,5,6],
                     [7,8,9]])
print(tensor)
print(tensor*tensor)  #normal çarpma
print(tensor @ tensor)#matrix çarpımı
print(torch.matmul(tensor, tensor))
print(tensor.T)
print(tensor.T.shape)