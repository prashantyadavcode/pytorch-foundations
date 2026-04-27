# Multi-layer model
# that can learn complex patterns

# Why we need more than linear? because linear cannot learn curves, patterns like XOR, complex decision boundaries
# Solution -> stack layers

# Basic NN Structure ->
# Input → Linear → ReLU → Linear → Output

# Building first NN

import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(1, 10), # input -> hidden
    nn.ReLU(), # activation function -> introduce non-linearity, lets model learn curves, complex mappings
    nn.Linear(10, 1) # hidden -> output
)

# in this we have inserted 1 input value, 10 neurons in hidden layer and 1 output layer


# Training Loop (same structure)


x = torch.linspace(-2, 2, 100)
y = x**2   # curve


loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(200):

    y_pred = model(x.view(-1, 1))

    loss = loss_fn(y_pred, y.view(-1, 1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(loss.item())


# Key Understanding ->
# Layer -> transformation
# Activation -> adds complexity
# Stack layers -> learn patterns

