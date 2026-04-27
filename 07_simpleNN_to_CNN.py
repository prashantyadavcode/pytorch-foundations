# Why CNN over Linear Layers?   

    # Previous Model - Flatten → Linear → ReLU → Linear
    # Problem - loses spatial information (pixel relationships)

    # CNN approach - Image → Convolution → ReLU → Pooling → Linear → Output
    # keeps structure intact

# Core CNN Concepts
    # Convolutional Layer - nn.Conv2d(input_channel (grayscale), feature_maps, filter)
    # ReLU - for non-linearity
    # MaxPooling - reduces size

import torch
import torchvision
import torchvision.transforms as transforms

# transform images → tensors
transform = transforms.ToTensor()

# load dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

# create dataloader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

# Build CNN Model

# Model - Image → Feature Extraction → Compression → Classification
# More concretely - (1,28,28) → Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → Linear → Output(10)

import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(1, 16, 3), # (1, 28, 28) -> (16, 26, 26)
    nn.ReLU(),
    nn.MaxPool2d(2), # -> (16, 13, 13)

    nn.Conv2d(16, 32, 3), # -> (32, 11, 11)
    nn.ReLU(),
    nn.MaxPool2d(2), # (32, 5, 5)

    nn.Flatten(),
    nn.Linear(32*5*5, 10)
)

# Each layer is learning ->
# Conv1 - edges, lines
# Conv2 - shapes, patterns
# Linear - classification

# Mental Model - Pixels → Edges → Shapes → Patterns → Digit

# Shape Flow (IMPORTANT)

# Layer -> Output Shape
    # Input -> (1, 28, 28)
    # Conv1 -> (16, 26, 26)
    # Pool -> (16, 13, 13)
    # Conv2 -> (32, 11, 11)
    # Pool -> (32, 5, 5)
    # Flatten -> (800, )
    # Output -> (10, )

# Why increasing Channels? 1 -> 16 -> 32
# more channels = more features learned

# Why skrinking size? 28 -> 26 -> 13 -> 11 -> 5
# reduce computation + focus on key information

# This model is a feature extractor (Conv layers) + classifier (Linear layer)



# Training Loop (same as before)

import torch

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(3):
    for images, labels in train_loader:

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch}, Loss: {loss.item()}')


# Accuracy Check

correct = 0
total = 0

with torch.no_grad():
    for images, labels in train_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: ', 100 * correct / total)


