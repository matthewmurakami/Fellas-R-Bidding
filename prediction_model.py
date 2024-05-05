import torch
import torch.nn as nn
import torch.optim as optim
# from torchvision import datasets, transforms
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

        self.linear_common = nn.Sequential(
            self.conv1,
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            self.conv2,
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
        )
        self.linear_actor = nn.Sequential(
            nn.Linear(1152, 18),  # Adjust this line based on actual output size
        )
        self.linear_critic = nn.Sequential(
            nn.Linear(1152, 1),  # Adjust this line based on actual output size
        )
        self.loss_fn = nn.HuberLoss()
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x):
        common = self.linear_common(x)
        action_probs = self.linear_actor(common)
        critic_value = self.linear_critic(common)
        return action_probs, critic_value