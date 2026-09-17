import torch
tensor = torch.arange(0,50,5)
print(tensor)
print(tensor.min())
print(tensor.max())
print(tensor.sum())
print(tensor.median())

print(tensor.dtype)
tensor = tensor.type(torch.float)
print(tensor.mean())

print(tensor.argmax())
print(tensor.argmin())

# manipulating tensors: reshaping, stacking, squeezing, unsqueezing
x = torch.arange(1,10,1)
print(x.shape)
x_reshaped = x.reshape(1,9)
print(x_reshaped)
print(x_reshaped.shape)

#stacking
print(torch.stack([x,x,x,x], dim=0))
print(torch.stack([x,x,x,x], dim=1))

#squeeze & unsqueeze
x = torch.tensor([[[1,2,3],[4,5,6]]])
print(x)
print(x.squeeze())
print(x.unsqueeze(dim=0))

#permute
x = torch.rand(size = (224,224,3))
print(x.shape)
x_permuted = x.permute(2,0,1) #axis 0->1, 1->2, 2->0
print(x_permuted.shape)

#indexing & slicing
a = torch.arange(1,10,1)
print(a)
print(a.reshape(1,3,3))
print(a[0])
print(a[:, 0])
print(a[:, :, 1])

#random seed
torch.manual_seed(seed=42) 
random_tensor = torch.rand(3, 4)
random_tensor