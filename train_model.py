import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from my_agent import process_saved_dir
import numpy as np
from allocation_model import AllocationNetwork
from prediction_model import PredictionNetwork
from sklearn.model_selection import train_test_split
from agt_server.agents.base_agents.lsvm_agent import MyLSVMAgent
from agt_server.local_games.lsvm_arena import LSVMArena
from agt_server.agents.test_agents.lsvm.min_bidder.my_agent import MinBidAgent
from agt_server.agents.test_agents.lsvm.jump_bidder.jump_bidder import JumpBidder
from agt_server.agents.test_agents.lsvm.truthful_bidder.my_agent import TruthfulBidder
from my_agent import MyAgent
import time
from torchsummary import summary


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


"""def train(dataloader, model):
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
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += (y - pred).absolute().sum()  
    test_loss /= num_batches
    # correct /= size
    correct /= size*18
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test Error: \n Prediction Error: {(correct):>0.001f}, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    datax, datay = process_saved_dir("saved_games")

    tensor_x = torch.Tensor(datax) # transform to torch tensor
    tensor_y = torch.Tensor(datay)

    # train-test split for evaluation of the model
    X_train, X_test, y_train, y_test = train_test_split(tensor_x, tensor_y, train_size=0.7, shuffle=True)
    
    # set up DataLoader for training set
    loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=32)

    model = PredictionNetwork()
    model = model.to(device)


    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(loader, model)
        test(loader, model)
    
    print("Done!")

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")"""


# Genetic algorithm parameters
POPULATION_SIZE = 10
MUTATION_RATE = 0.1
NUM_GENERATIONS = 5

def compute_fitness(model, train_loader, test_loader):

    # model.train()
    # for epoch in range(5):
    #     for data, target in train_loader:
    #         model.optimizer.zero_grad()
    #         output = model(data)
    #         loss = model.loss_fn(output, target[1])
    #         loss.backward()
    #         model.optimizer.step()

    # model.eval()
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data, target in test_loader:
    #         output = model(data)
    #         _, predicted = torch.max(output.data, 1)
    #         total += target[1].size(0)
    #         correct += (predicted == target[1]).sum().item()

    # accuracy = correct / total
    # return accuracy
    pass

# Initialize genetic algorithm parameters
def initialize_population(names):
    population = []
    for i in range(POPULATION_SIZE):
        model = PredictionNetwork(name=names[i])
        print(summary(model, (1,3,6)))
        exit()
        population.append(model)
    return population

# Crossover operator: Single-point crossover
def crossover(parent1, parent2):
    child1 = PredictionNetwork()
    child2 = PredictionNetwork()
    child1.conv1.weight.data = torch.cat((parent1.conv1.weight.data[:16], parent2.conv1.weight.data[16:]), dim=0)
    child2.conv1.weight.data = torch.cat((parent2.conv1.weight.data[:16], parent1.conv1.weight.data[16:]), dim=0)
    return child1, child2

# Mutation operator: Random mutation
def mutate(model):
    for param in model.parameters():
        if torch.rand(1).item() < MUTATION_RATE:
            param.data += torch.randn_like(param.data) * 0.1  # Adding Gaussian noise with std=0.1
    return model


def quickSort(array):
    if len(array)> 1:
        pivot=array.pop()
        grtr_lst, equal_lst, smlr_lst = [], [pivot], []
        for item in array:
            if item[1] == pivot[1]:
                equal_lst.append(item)
            elif item[1] > pivot[1]:
                grtr_lst.append(item)
            else:
                smlr_lst.append(item)
        return (quickSort(smlr_lst) + equal_lst + quickSort(grtr_lst))
    else:
        return array

MODELS_PATH = "models/"

if __name__ == "__main__":

    population = None

    for generation in range(NUM_GENERATIONS):
        print("Generation:", generation + 1)
        best_accuracy = 0
        best_individual = None

        names = ["Generation: " + str(generation)+ " Bot: " + str(i) for i in range(POPULATION_SIZE)]
        # Initialize population
        if population is None:
            population = initialize_population(names)
        
        population_players = [MyAgent(name) for name in names]

        for i, player in enumerate(population):
            torch.save(player.state_dict(), MODELS_PATH + "{names[i]}")
    
        default_players=[
            MinBidAgent("Min Bidder"),
            JumpBidder("Jump Bidder"), 
            TruthfulBidder("Truthful Bidder")]


        arena = LSVMArena(
            num_cycles_per_player = 10,
            timeout=1,
            local_save_path="saved_games",
            players=population_players,
        )
        
        arena.run()

        datax, datay = process_saved_dir("saved_games")
        tensor_x = torch.Tensor(datax) 
        tensor_y = torch.Tensor(datay)

        player_datax = []
        player_datay = []
        for i in range(tensor_x.shape[0]):
            name = tensor_y[i][0]
            player_datax[name] += tensor_x[i]
            player_datay[name] += tensor_y[i][1]

        # Compute fitness for each individual
        fitness_arr = []
        for individual in population:
            name = individual.name
            # X_train, X_test, y_train, y_test = train_test_split(player_datax[name], player_datay[name], train_size=0.7, shuffle=True)
        
            # train_loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=32)
            # test_loader = DataLoader(list(zip(X_test, y_test)), shuffle=True, batch_size=32)

            # fitness = compute_fitness(individual, train_loader, test_loader)

            elo = player_datay[name]

            for i in range(len(elo)):
                elo[i] = (1/(len(elo) - i)) * elo[i]
            fitness = sum(elo)/len(elo)

            if fitness > best_accuracy:
                best_accuracy = fitness
                best_individual = individual
            
            fitness_arr.append((individual, fitness))
        
        fitness_arr = quickSort(fitness_arr)

        print("Best accuracy in generation", generation + 1, ":", best_accuracy)
        print("Best individual:", best_individual)

        next_generation = []

        # Select top individuals for next generation
        selected_individuals = fitness_arr[:POPULATION_SIZE // 2]
        selected_individuals = [i[0] in selected_individuals]

        

        # Crossover and mutation
        for i in range(0, len(selected_individuals), 2):
            parent1 = selected_individuals[i]
            parent2 = selected_individuals[i + 1]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            next_generation.extend([child1, child2])

        population = next_generation