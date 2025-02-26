import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageClassificationCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassificationCNN, self).__init__()

        # Convolutional layers - removed one layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Dropout
        self.dropout = nn.Dropout(0.3)

        # Calculate size after convolutions and pooling
        # Assuming input size of 224x224
        # After 3 pooling layers: 224 -> 112 -> 56 -> 28
        self.fc_input_size = 128 * 28 * 28

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)  # Reduced size
        self.fc2 = nn.Linear(512, num_classes)  # Removed one FC layer

    def forward(self, x):
        # Convolutional blocks with activation, batch norm and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 224 -> 112
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 112 -> 56
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 56 -> 28

        # Flatten
        x = x.view(-1, self.fc_input_size)

        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x