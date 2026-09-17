import torch
print(torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu" # other than mac
print(device)

if torch.cuda.is_available():
    device = "cuda" # Use NVIDIA GPU 
    print(device)
elif torch.backends.mps.is_available():
    device = "mps" # Use Apple Silicon GPU
    print(device) 
else:
    device = "cpu" # Default to CPU if no GPU is available
    print(device)

tensor = torch.tensor([1, 2, 3])
print(tensor.device)

# 1) Manually handle everything
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu.device)

#2) second option would be to use the context manager that came after 2.x versions
with torch.device(device):
    # All tensors created in this block will be on device
    tensor2 = torch.tensor([1,2,3])
    layer = torch.nn.Linear(20, 30) # we haven't covered layers yet but this will be created on gpu as well
print(tensor2.device)

print(layer.weight.device)

#3) third option would be to use global definition of device for pytorch that came after 2.x versions
torch.set_default_device(device)
tensor3 = torch.tensor([1,2,3])
layer2 = torch.nn.Linear(20, 30)
print(tensor3.device)
print(layer2.weight.device)
print(torch.set_default_device("cpu"))