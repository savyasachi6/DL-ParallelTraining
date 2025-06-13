import torch.nn as nn
from torchvision.models import resnet50

class SimpleResNet(nn.Module):
    """
    A simplified ResNet model based on ResNet50 with custom fully connected layers.
    
    Args:
        num_classes (int): Number of output classes (default: 10 for CIFAR-10)
    """
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(2048, 512)
        
        self.dropout = nn.Dropout(0.3)  # Dropout to prevent overfitting
        self.fc1 = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)  # Softmax activation for multi-class classification
    
    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc1(x)
        x = self.softmax(x)  # Apply Softmax activation
        return x
