import torch

# inputs
x = torch.tensor([1.0, 2.0, 3.0, 4.0])

# outputs (y = 2x + 1)
y = torch.tensor([4.0, 8.0, 12.0, 16.0])

# Intializing parameters
w = torch.tensor(0.0, requires_grad = True)
b = torch.tensor(0.0, requires_grad = True)

# Training Loop (Core of ML)
learning_rate = 0.01

for epoch in range(100):

    # Forward pass
    y_pred = w * x + b

    # Loss (Mean Squared Error)
    loss = ((y_pred - y) ** 2).mean()

    # Backward pass
    loss.backward()

    # Update weights
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Reset gradient
    w.grad.zero_()
    b.grad.zero_()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

print(w.item(), b.item())


# Upgrade (Cleaner Pytorch way)

# Input → Prediction → Error → Gradient → Update → Repeat

# Forward -> make predictions
# Loss -> measure error
# Backward -> compute gradients
# Update -> improve model

import torch.nn as nn

model = nn.Linear(1, 1) # automatically handles weights and bias

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(500):

    y_pred = model(x.view(-1, 1))

    loss = loss_fn(y_pred, y.view(-1, 1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(loss.item())

# this block is foundation of ALL deep learning. doesn't matter if its CNN, NLP or Transformers. The loop stay the same




