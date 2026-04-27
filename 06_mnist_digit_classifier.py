# MNIST Digit Classifier
# Build a model that can classify handwritten digits (0-9)

# What is MNIST?
# 28 x 28 grayscale images
# digits: 0 -> 9
# total classes = 10

# Input = image
# Output = digit label

# Problem Type -> classification, not regression
# so instead of y = wx + b, now output = probs for 10 classes

# Model does - Image → Features → Scores → Predicted digit

# Load dataset
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.ToTensor() # ToTensor() -> converts image to tensor

train_dataset = torchvision.datasets.MNIST(
    root = './data',
    train = True,
    transform = transform,
    download = True
)

train_loader = torch.utils.data.DataLoader( # DataLoader -> gives batches (very important)
    dataset = train_dataset,
    batch_size = 64,
    shuffle = True
)

# Build Model

import torch.nn as nn

model = nn.Sequential(
    nn.Flatten(), # 28 x 28 -> 784
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Loss + Optimizer 

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# Why CrossEntropy?
# because multi-class classification, handles probabilities internally

# Training Loop
for epoch in range(3):
    for images, labels in train_loader:
        outputs = model(images)

        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch}, Loss: {loss.item()}')

