import torch

# ============================================= #
#              Initializing Tensor              #
# ============================================= #

device = "cuda" if torch.cuda.is_available() else "cpu"

#intializing with data
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype = torch.float32, device = device, requires_grad=True)

# other common initialization methods
x = torch.empty(size = (3, 3)) # creates a 3x3 tensor (matrix) that is unitialized
x = torch.zeros((3, 3))
x = torch.rand((3,3)) # random 3 by 3 array
x = torch.eye(5, 5) # identity matrix
x = torch.arange(start=0, end=5, step=1)
x = torch.arange(10) # integers from 0 to 9
x = torch.linspace(start=0.1, end=1, steps=19)
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1) #creates 1x5 and makes them follow the distribution
x = torch.empty(size=(1, 5)).uniform_(0,1)
x = torch.diag(torch.ones(3)) #creates a 3x3 identity matrix

# how to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)
print(tensor.bool()) # boolean
print(tensor.short()) # int16
print(tensor.long()) # int64 (Important)
print(tensor.half()) # float16
print(tensor.float()) # float32 (Important)
print(tensor.double()) # float64

# array to tensor conversion and vice-versa
import numpy as np
np_array = np.zeros((5,5))
tesnor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()