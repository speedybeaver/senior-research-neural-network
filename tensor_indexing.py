import torch

# ============================================= #
#                Tensor Indexing                #
# ============================================= #

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

#print(x[0].shape) # x[0,:] or all features of the first batch

#print(x[:, 0].shape) # get the first feature of all batches

#print(x[2, 0:10]) # 0:10 --> {0, 1, 2, ..., 9}

x[0, 0] = 100

# fancy indexing

x = torch.arange(10)
indices = [2, 5, 8]

x = torch.rand((3,5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
#print(x[rows, cols])

# more advanced indexing
x = torch.arange(10)
#print(x[(x < 2) | (x > 8)])

# useful indexing
print(torch.where(x > 5, x, x*2)) # double the numbers less than five
print(torch.tensor([0,0,1,2,2,3,2,3,4,4]).unique())
print(x.numel()) # prints the number of elements