import glob
import torch, sys, os
import torch.nn as nn
import torch.optim as optim
# from torchvision import datasets, transforms
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
from my_agent import MyAgent, TrainingAgent
import time
from torchsummary import summary
import random
import copy
import re

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Genetic algorithm parameters
POPULATION_SIZE = 10
MUTATION_RATE = 0.1
NUM_GENERATIONS = 5
MODELS_PATH = "models/"
DISCOUNT_FACTOR = 0.95


def train(model, input, reward):
    discount_rewards = discount(np.full((input.shape[0]),reward))
    print(discount_rewards)
    action_probs, critic_value = individual.forward(input)

    diff = discount_rewards - critic_value
    actor_loss = -actor_history * diff
    critic_loss = self.model.loss_fn(critic_history, discount_rewards)
    loss_value = sum(actor_loss) + sum(critic_loss)

    loss_value.backward()
    self.model.optimizer.step()
    
    self.actor_history = []
    self.critic_value_history = []

    # with torch.GradientTape() as tape:
    #     actor_losses = []
    #     critic_losses = []

    #     # print("actor history length:", len(actor_history))
    #     # print("critic history length:", len(critic_history))
    #     # print("discount rewards  length:", len(discount_rewards))

    #     # for actor, critic, reward in zip(actor_history, critic_histort, discount_rewards):
    #     for i in range (len(actor_history)):
    #         actor = actor_history[i]
    #         critic = critic_history[i]
    #         reward = discount_rewards[i]
            
    #         actor = torch.cast(torch.reshape(actor, [-1]), torch.float32)
    #         critic = torch.cast(critic, torch.float32)
    #         reward = torch.cast(reward, 
    #                             torch.float32)

    #         diff = reward - critic

    #         # Trying to make (6x3) * (2) compatitable
    #         #actor = tf.reshape(actor, [-1]) # flatten
    #         # print(critic_history)
    #         # print(critic)
    #         # print(diff[i])

    #         # print("Actor shape:", actor.shape)
    #         # print("Diff shape:", diff.shape)
    #         # print("Reward shape:", reward.shape)

    #         # print("DEBUG ============> -actor:", -actor)
    #         # print("DEBUG ============>  diff: ", diff[i])

    #         actor_losses.append(-actor * diff[i])  # actor loss

    #         # The critic must be updated so that it predicts a better estimate of the future rewards.
    #         critic_losses.append(self.huber_loss(torch.expand_dims(critic, 0), torch.expand_dims(reward, 0)))

    #     # Backpropagation
    #     loss_value = sum(actor_losses) + sum(critic_losses)
    #     grads = tape.gradient(loss_value, self.model.trainable_variables)
    #     self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    #     self.critic_value_history.clear()

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

def discount(rewards):
    if len(rewards) == 1:
        return rewards

    indices = np.arange(len(rewards))
    total = rewards[0]
    total = total + np.sum(rewards[1:] * np.power(DISCOUNT_FACTOR, indices[1:]))
    return np.concatenate((np.array([total]), DISCOUNT_FACTOR(rewards[1:])))

# Initialize genetic algorithm parameters
def initialize_population(names):
    population = []
    for i in range(POPULATION_SIZE):
        model = PredictionNetwork(name=names[i])
        # print(summary(model, (1,3,6)))
        # exit()
        population.append(model)
    return population

# Crossover operator: Single-point crossover
def crossover(parent1, parent2, name1, name2):
    child1 = PredictionNetwork(name=name1)
    child2 = PredictionNetwork(name=name2)
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
    
def proccess_output(file_path):
    output = {}
    elo_section = False
    with open(file_path) as file:
        for line in file:
            line = line.strip()
            if not elo_section:
                # auction_num = re.match("Auction (\d+):.*", line)
                # if auction_num:
                #    continue
                utility = re.match("^([^\s]+) got a final utility of (-?\d*[.,]?\d*)$", line)
                if utility:
                    if utility.group(1) not in output:
                        output[utility.group(1)] = [float(utility.group(2))]
                    else:
                        output[utility.group(1)].append(float(utility.group(2)))
                    continue
                if re.match("Agent Name  Final Score   ELO", line):
                    elo_section = True
                    continue
            # else:
            #     output = re.match("^\d+[\s]+([^\s]+)[\s]+(\d*[.,]?\d*)[\s]+(\d+)$", line)

            #     output[elo.group(1)] = int(elo.group(3))
    return output
            



if __name__ == "__main__":

    population = None
    names = None
    
    for generation in range(NUM_GENERATIONS):
        files = glob.glob('saved_games/*')
        for f in files:
            os.remove(f)
        print("Generation:", generation + 1)
        best_accuracy = 0
        best_individual = None

       
        # Initialize population
        if population is None:
            names = ["Generation_" + str(generation)+ "_Bot_" + str(i) for i in range(POPULATION_SIZE)]
            population = initialize_population(names)
        
        for i, player in enumerate(population):
            torch.save(player.state_dict(), MODELS_PATH + names[i] + '.pth')
        
        population_players = [MyAgent(name) for name in names]

        default_players=[
            MinBidAgent("Min_Bidder"),
            JumpBidder("Jump_Bidder"), 
            TruthfulBidder("Truthful_Bidder"),
            TrainingAgent("Training_Agent"),]

        bidders = copy.deepcopy(population_players)
        bidders.extend(default_players)

        arena = LSVMArena(
            num_cycles_per_player = 1,
            timeout=1,
            local_save_path="saved_games",
            players=bidders,
        )   

        with open('output.txt', 'w') as sys.stdout:
            arena.run()
        sys.stdout = sys.__stdout__

        data = process_saved_dir("saved_games")
        score = proccess_output('output.txt')
        
            

        # Compute fitness for each individual
        fitness_arr = []
        for individual in population:
            name = individual.name
            # X_train, X_test, y_train, y_test = train_test_split(player_datax[name], player_datay[name], train_size=0.7, shuffle=True)
        
            # train_loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=32)
            # test_loader = DataLoader(list(zip(X_test, y_test)), shuffle=True, batch_size=32)

            # fitness = compute_fitness(individual, train_loader, test_loader)
            print(name)
            print(len(data[name]))
            print(len(score[name]))
            continue
            fitness = score[name]
            # train(individual, data[name], fitness)

            


            if fitness > best_accuracy:
                best_accuracy = fitness
                best_individual = individual
            
            fitness_arr.append((individual, fitness))
        exit()
        
        fitness_arr = quickSort(fitness_arr)

        print("Best accuracy in generation", generation + 1, ":", best_accuracy)
        print("Best individual:", best_individual.name)

        next_generation = []

        # Select top individuals for next generation
        selected_individuals = fitness_arr[-POPULATION_SIZE // 2:]
        selected_individuals = [i[0] for i in selected_individuals]

        # Crossover and mutation
        names = ["Generation_" + str(generation+1)+ "_Bot_" + str(i) for i in range(POPULATION_SIZE)]
        for i in range(0, POPULATION_SIZE, 2):
            parent1, parent2 = random.sample(selected_individuals,2)
            child1, child2 = crossover(parent1, parent2, names[i], names[i+1])
            child1 = mutate(child1)
            child2 = mutate(child2)
            next_generation.extend([child1, child2])

        population = next_generation