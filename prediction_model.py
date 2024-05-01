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

# class PredictionNetwork(nn.Module):    
#     def __init__(self, name=None):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.name = name

#         # self.linear_relu_stack = nn.Sequential(
#         #     self.conv1,
#         #     nn.MaxPool2d(kernel_size=3, stride=1),
#         #     self.conv2,
#         #     nn.MaxPool2d(kernel_size=3, stride=1),
            
#         #     nn.Linear(64 * 18, 18),
#         #     nn.ReLU()
#         # )
#         self.linear_relu_stack = nn.Sequential(
#             self.conv1,
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # Added padding
#             self.conv2,
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # Added padding
#             nn.Flatten(),
#             nn.Linear(64 * 4 * 6, 18),  # Adjusted linear layer to match the new flat size
#             nn.ReLU()
#         )
#         self.loss_fn = nn.NLLLoss()
#         self.optimizer = optim.Adam(self.parameters())

#     def forward(self, x):
#         # Input Shape: torch.Size([2, 1, 3, 6])
#         logits = self.linear_relu_stack(x)
#         return logits
class PredictionNetwork(nn.Module):    
    def __init__(self, name=None):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.name = name

        self.linear_relu_stack = nn.Sequential(
            self.conv1,
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            self.conv2,
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Flatten(start_dim=0),
            nn.Linear(1152, 18),  # Adjust this line based on actual output size
            nn.ReLU()
        )
        self.loss_fn = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x):
        # x = self.conv1(x)
        # # print("After conv1:", x.shape)
        # x = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(x)
        # # print("After pool1:", x.shape)
        # x = self.conv2(x)
        # # print("After conv2:", x.shape)
        # x = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(x)
        # # print("After pool2:", x.shape)
        # x = x.flatten(start_dim=0)
        # # print("After flatten:", x.shape)
        # x = nn.Linear(1152, 18)(x)  # Adjust based on actual size
        # # print("After linear:", x.shape)
        # x = nn.ReLU()(x)
        x = self.linear_relu_stack(x)
        return x