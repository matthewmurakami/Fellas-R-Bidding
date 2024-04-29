import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from my_agent import process_saved_dir
import numpy as np
from allocation_model import AllocationNetwork
from sklearn.model_selection import train_test_split

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
    for i, batch in enumerate(dataloader):
        X, y = batch
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = model.loss_fn(pred, y)

        # Backpropagation
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        if i % 100 == 0:
            loss, current = loss.item(), (i + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += model.loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    datax, datay = process_saved_dir("saved_games")

    tensor_x = torch.Tensor(datax) # transform to torch tensor
    tensor_y = torch.Tensor(datay)

    # train-test split for evaluation of the model
    X_train, X_test, y_train, y_test = train_test_split(tensor_x, tensor_y, train_size=0.7, shuffle=True)
    
    # set up DataLoader for training set
    loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=32)

    model = AllocationNetwork()
    model = model.to(device)


    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(loader, model)
        test(loader, model)
    
    print("Done!")

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")



    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    
    # Read data, convert to NumPy arrays
    data = pd.read_csv("sonar.csv", header=None)
    X = data.iloc[:, 0:60].values
    y = data.iloc[:, 60].values
    
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    
    # convert into PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    
    # train-test split for evaluation of the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
    
    # set up DataLoader for training set
    loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=16)
    
    # create model
    model = nn.Sequential(
        nn.Linear(60, 60),
        nn.ReLU(),
        nn.Linear(60, 30),
        nn.ReLU(),
        nn.Linear(30, 1),
        nn.Sigmoid()
    )
    
    # Train the model
    n_epochs = 200
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.train()
    for epoch in range(n_epochs):
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # evaluate accuracy after training
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    acc = float(acc)
    print("Model accuracy: %.2f%%" % (acc*100))
    """