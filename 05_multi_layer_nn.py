import torch.nn as nn

model = nn.Linear(1, 1) # automatically handles weights and bias

loss_fn = nn.MSELoss()
optimzer = torch.optim.SGD(model.paramters(), lr = 0.01)

for epoch in range(100):

    y_pred = model(x.view(-1, 1))

    loss = loss_fn(y_pred, y.view(-1, 1))

    optimzer.zero_grad()
    loss.backward()
    optimzer.step()

    if epoch % 10 == 0:
        print(loss.item())