import torch
import torch.nn as nn
import torch.optim as optim
# from torchvision import datasets, transformss
from torch.utils.data import DataLoader

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class AllocationNetwork(nn.Module):    
    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(18, 500),
        )
        self.softmax_layer = nn.Softmax()
        self.loss_fn = nn.NLLLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)

        temp = torch.zeros((logits.shape[0],1)).to(device)
        logits = torch.hstack((logits, temp))

        logits = self.softmax_layer(logits)
        return logits