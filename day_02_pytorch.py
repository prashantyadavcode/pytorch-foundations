# Tensor Operations + GPU

# Indexing (Access specific values)
import torch

x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(x[0]) # first element
print(x[1][2]) # access 6
print(x[1, 2]) # cleaner way to access 6

# Slicing (Extract Parts)

print(x[:, 1]) # all rows, 2nd column -> [2, 5, 8]
print(x[0:2, :]) # first 2 rows
print(x[:, :2]) # first 2 columns

# Broadcasting 

a = torch.rand(3, 4)
b = torch.rand(4)

c = a + b

print(c)

# (here if b = torch.rand(3) in that case its invalid case as b is treated as (1, 3) and it mismatch last dimension)


# GPU Basics (.to(device))
# this is where pytorch beats numpy

# Step 1 - check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Step 2 - Move tensor to device
# (all tensors in a operation must be on SAME DEVICE)

x = torch.rand(3, 3)

x = x.to(device)
print(x)

a = torch.rand(3, 3).to("cpu")
b = torch.rand(3, 3)

c = a + b  


# Practice Session

# Step 1

x = torch.rand(4, 5)

# Step 2

# print:
# 2nd row
# 3rd column
# submatrix (first 2 rows, first 3 columns)
print(x[1])
print(x[:, 2])
print(x[0:2, 0:3])


# Step 3

y = torch.rand(5)

# Step 4

z = x + y   # broadcasting

# Step 5
# move everything to GPU (if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

x = x.to(device)
y = y.to(device)
z = z.to(device)

# Step 6
# print device of z
print(z.device)


# END
