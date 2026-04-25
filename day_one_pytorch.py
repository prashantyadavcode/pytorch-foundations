# Check the version of torch and cuda

import torch
print(torch.__version__)
print(torch.cuda.is_available())

# First Tensor Program

# Create a tensor
x = torch.tensor([1, 2, 3])
print(x)

# Random tensor
y = torch.rand(3, 3)
print(y)

# Zero & ones
z = torch.zeros(2, 2)
o = torch.ones(2, 2)

print(z)
print(o)

# Tensor Operations

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(a + b)
print(a * b)
print(torch.dot(a, b))

# Tensor shape

print(y.shape)

# Reshape

k = torch.rand(2, 6)
r = k.view(3, 4)
print(r.shape)

# Data type

print(x.dtype)
# can also change it using x = torch.tensor([1, 2, 3], dtype = torch.float32)


# NumPy to Pytorch Conversion

import numpy as np
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)

# Back to NumPy
back_to_numpy = tensor.numpy()


# Mini Practice

# * Create a 4×4 random tensor
# * Convert it to shape (2, 8)
# * Add another tensor
# * Print dtype

m = torch.rand(4, 4)
f = m.view(2, 8)
q = torch.tensor([1, 2, 3, 4])
t = m + q

print("Original:\n", m)
print("Reshaped:\n", f)
print("Result:\n", t)
print("dtype:", t.dtype)
