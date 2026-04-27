import torch

# Enable Gradient Tracking

x = torch.tensor(2.0, requires_grad = True)
y = x ** 2

print(y)

# Backpropagation 

y.backward()
print(x.grad)

# because y = sq(x) , dy/dx = 2x, at x = 2 -> 4

# Computational Graph (Important Concept)
# pytorch builds a graph like -
# x -> square -> y

# Rule 1: Gradients accumulate

y = x**2
y.backward()

# to fix this we will use 
# x.grad.zero_()

y = x**2
y.backward()

print(x.grad) # 12, not 4


# Rule 2: Use .detach() when needed

# Stop tracking:
y = x.detach()


# Rule 3: No grad mode 
with torch.no_grad():
    y = x ** 2



# Mini Example

a = torch.tensor(1.0, requires_grad = True)
b = 3 + a ** 2 + 2 * a + 1

b.backward()
print(a.grad)


# Practice Session

# Step 1
c = torch.tensor(3.0, requires_grad=True)

# Step 2
d = c**3 + 2*c**2 + c

# Step 3
# compute gradient
d.backward()

# Step 4
# print gradient
print(c.grad)

# END