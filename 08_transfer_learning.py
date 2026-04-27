# Transfer Learning
# instead of training from scratch we use a pretrained model and adapt it

# Why transfer learning?
# training from scratch - needs lots of data, takes time and harder to optimize

# Basically, Transfer Learning - use knowledge leanred on huge dataset -> adpat to your task
# models trained on millions of images
# already understands edges, textures, shapes

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

# Load Pretrained Model - ResNet18

import torch
import torch.nn as nn
import torchvision.models as models

model = models.resnet18(pretrained = True)

# a powerful CNN already trained
# feature extractor ready


# Modify Final Layer - original classes (ImageNet) - 1000, we need 10 for MNIST

model.fc = nn.Linear(model.fc.in_features, 10)


# Freeze Layers (Important)

for param in model.parameters():
    param.requires_grad = False

# prevents retraining entire network


# Train only last layer
for param in model.fc.parameters():
    param.requires_grad = True



# Adjust Input Size - ResNet expects (3, 224, 224) but MNIST is (1, 28, 28)
# Fix with transforms

import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

# Training Loop
loss_fn = nn.CrossEntropyLoss()
optimizier = torch.optim.Adam(model.fc.parameters(), lr = 0.001)

for epoch in range(3):
    for images, labels in train_loader:

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizier.zero_grad()
        loss.backward()
        optimizier.step()
    
    print(loss.item())