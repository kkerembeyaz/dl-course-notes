import torch
import numpy as np

#1)Pytorch Datatypes - Tensors
print(torch.tensor(10))
print(torch.tensor([10,20]))

scalar = torch.tensor(10)
print(type(scalar))

print(scalar +10 - 5)
print(scalar*3/2)
print(scalar.dim())
print(scalar.shape)
print(scalar.item())
print(type(scalar.item()))

#2)Vectors
vector = torch.tensor([2,3])
print(vector.dim())
print(vector.shape)
# count of [ in order to figure out number of dimensions
# count elements in order to figure out shape
matrix = torch.tensor([[1,2],[3,4]])
print(matrix.ndim)
print(matrix.shape)

float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # default or custom?
                               device=None, # cpu, gpu?
                               requires_grad=False) # If autograd should record operations on the returned tensor