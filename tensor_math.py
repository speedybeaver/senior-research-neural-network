import torch

# ============================================= #
#                  Tensor Math                  #
# ============================================= #

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

# addition
z = torch.true_divide(y, x)
print(z)

# useful tensor operations
sum_x = torch.sum(x, dim=0)
print(sum_x)

values, indices = torch.max(x, dim=0)
print(f"Maximum is {values} at {indices}")

values1, indices1 = torch.min(x, dim=0)
print(f"Minimum is {values1} at {indices1}")

mean_x = torch.mean(x.float(), dim=0)
print(f"mean is {mean_x}")

sorted_y, indices = torch.sort(y, dim=0, descending=False) # sort in ascending order
print(sorted_y)

z = torch.clamp(x, min=0, max=2)
print(z)