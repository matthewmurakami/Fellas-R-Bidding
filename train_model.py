import torch
from torch import nn
from torch.utils.data import DataLoader
from my_agent import process_saved_dir
import numpy as np

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


def train(dataloader, model):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = model.loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    # epochs = 5
    # for t in range(epochs):
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     train(train_dataloader, model)
    #     test(test_dataloader, model)
    
    # print("Done!")

    # torch.save(model.state_dict(), "model.pth")
    # print("Saved PyTorch Model State to model.pth")
    data = np.fromiter(process_saved_dir("saved_games"), float)
    print(data)