import torch
import numpy as np

random_array = np.random.randn(3,5)
print(random_array)
print(type(random_array))
print(random_array.ndim)

random_matrix = torch.rand(size =(3,5))
print(random_matrix)
print(type(random_matrix))
print(random_array.ndim)

random_image_tensor = torch.rand(size=(224,224,3))
print(random_image_tensor)

vector_with_arange = torch.arange(start=0, end=10, step=1)
print(vector_with_arange)

# we can create zeros and ones if we need some for matrix multiplication or anything else
zeros = torch.zeros(size=(3, 2))
zeros
tensor([[0., 0.],
        [0., 0.],
        [0., 0.]])

ones = torch.ones(size=(3, 4))
ones
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])

# basic math operations
tensor = torch.tensor([1, 2, 3])
tensor + 15=tensor([16, 17, 18])
tensor * 12=tensor([12, 24, 36])
tensor * 3.14=tensor([3.1400, 6.2800, 9.4200])
tensor = tensor + 10=tensor([11, 12, 13])

# we also have native methods for that
tensor.subtract(10)
tensor([1, 2, 3])
tensor
tensor([11, 12, 13])
tensor = tensor.subtract(10)
tensor
tensor([1, 2, 3])
tensor.multiply(20)
tensor([20, 40, 60])
tensor.add(20)
tensor([21, 22, 23])
tensor.divide(2)
tensor([0.5000, 1.0000, 1.5000])
# can we only operate with scalars?
tensor * tensor
tensor([1, 4, 9])
tensor + tensor
tensor([2, 4, 6])
tensor - tensor
tensor([0, 0, 0])
tensor / tensor
tensor([1., 1., 1.])
# beware, in this case shapes should make sense
tensor2 = torch.tensor([4,5])
tensor + tensor2 #it doesn't work.