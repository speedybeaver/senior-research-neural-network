import torch

# ============================================= #
 #              Tensor Reshaping                #
# ============================================= #

x = torch.arange(9)

x_3x3 = x.view(3,3) #contiguous tensors aka its stored 
x_3x3 = x.reshape(3,3) #more versatile, but view could have better performance

y = x_3x3.t() # transpose the 3x3, jumping different amounts of steps in memory, NOT contiguous

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))

#print(torch.cat((x1, x2), dim=0)) # concatenate along the first dimension (x)
z = x1.view(-1) #flattens the entire array
print(z.shape)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)

z = x.permute(0, 2, 1) #
print(z.shape)

x = torch.arange(10) # [10]
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)

x = torch.arange(10).unsqueeze(0).unsqueeze(1) # 1x1x10

z = x.squeeze(1)
print(z.shape)