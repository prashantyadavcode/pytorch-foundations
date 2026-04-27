# BatchNorm and Dropout

# Batch Norm is used to normalize activation function to maintain stability
# Conv → BatchNorm → ReLU - before activation function

# DropOut - is used to reduce neurons at end to avoid overfitting and model be over confident about data


# an example of both:

import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(1, 16, 3),
    nn.BatchNorm2d(16), # Normalize features
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(16, 32, 3),
    nn.BatchNorm2d(32), # Normalize feature
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),

    nn.Linear(32*5*5, 128),
    nn.ReLU(),
    nn.Dropout(0.5), # Dropout half neurons to avoid overfitting

    nn.Linear(128, 10)
)

