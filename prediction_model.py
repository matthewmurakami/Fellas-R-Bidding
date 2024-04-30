import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class PredictionNetwork(nn.Module):    
    def __init__(self, name=None):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.name = name

        self.linear_relu_stack = nn.Sequential(
            self.conv1,
            # nn.MaxPool2d(kernel_size=3, stride=1),
            self.conv2,
            # nn.MaxPool2d(kernel_size=3, stride=1),
            
            nn.Linear(64 * 18, 18),
            nn.ReLU()
        )
        self.loss_fn = nn.NLLLoss()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
