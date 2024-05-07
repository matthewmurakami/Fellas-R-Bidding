from agt_server.agents.base_agents.lsvm_agent import MyLSVMAgent
from agt_server.local_games.lsvm_arena import LSVMArena
from agt_server.agents.test_agents.lsvm.min_bidder.my_agent import MinBidAgent
from agt_server.agents.test_agents.lsvm.jump_bidder.jump_bidder import JumpBidder
from agt_server.agents.test_agents.lsvm.truthful_bidder.my_agent import TruthfulBidder
from prediction_model import PredictionNetwork
import time
import os
import gzip
import json
import torch
import numpy as np
from prediction_model import PredictionNetwork
from torch import load, from_numpy
from itertools import combinations
import time
import sys

MODELS_PATH = "models/"
NAME = 'Abandon Ship'

class Original_Training_Agent(MyLSVMAgent):
    def setup(self):
        pass 
    
    def national_bidder_strategy(self): 
        return self.regional_bidder_strategy()

    def get_averages(self, val_matrix):
        avg_matrix = np.zeros((3,6))

        for iy, ix in np.ndindex(val_matrix.shape):
            neighbors = np.array(())
            if(iy - 1 >= 0):
                neighbors = np.append(neighbors, val_matrix[iy-1, ix])
            if(iy + 1 < 3):
                neighbors = np.append(neighbors, val_matrix[iy+1, ix])
            if(ix - 1 >= 0):
                neighbors = np.append(neighbors, val_matrix[iy, ix-1])
            if(ix + 1 < 6):
                neighbors = np.append(neighbors, val_matrix[iy, ix+1])
            avg_matrix[iy,ix] = np.average(neighbors)
        return avg_matrix

    def regional_bidder_strategy(self): 
        min_bids = self.get_min_bids()
        valuations = self.get_valuations() 
        goods = self.get_goods()
        bids = {} 

        val_matrix = self.get_valuation_as_array()

        avg_val_matrix = self.get_averages(val_matrix)

        max = np.max(avg_val_matrix)
        locX,loxY = np.where(avg_val_matrix == max)
        loc = (locX[0],loxY[0])

        for good in goods:
            good_val = valuations[good]
            good_min_bid = min_bids[good]

            if loc == self.get_goods_to_index()[good] or \
                (abs(loc[0] - self.get_goods_to_index()[good][0]) == 1 and loc[1] == self.get_goods_to_index()[good][1]) or \
                (abs(loc[1] - self.get_goods_to_index()[good][1]) == 1 and loc[0] == self.get_goods_to_index()[good][0]):   
            
                if good_min_bid <= good_val + 1:
                    bids[good] = good_min_bid
                else:
                    bids[good] = 0
            else:
                if good_min_bid <= good_val:
                    bids[good] = good_min_bid
                else:
                    bids[good] = 0

        return bids

    def get_bids(self):
        if self.is_national_bidder(): 
            return self.national_bidder_strategy()
        else: 
            return self.regional_bidder_strategy()
    
    def update(self):
        pass 

class Matt_Ultra_Greedy_Agent(MyLSVMAgent):
    def setup(self):
        pass

    def get_averages(self, val_matrix):
        avg_matrix = np.zeros((3, 6))
        for iy, ix in np.ndindex(val_matrix.shape):
            neighbors = []
            if iy - 1 >= 0:
                neighbors.append(val_matrix[iy-1, ix])
            if iy + 1 < 3:
                neighbors.append(val_matrix[iy+1, ix])
            if ix - 1 >= 0:
                neighbors.append(val_matrix[iy, ix-1])
            if ix + 1 < 6:
                neighbors.append(val_matrix[iy, ix+1])
            avg_matrix[iy, ix] = np.mean(neighbors) if neighbors else 0
        return avg_matrix

    def evaluate_bundle(self, bundle):
        valuations = self.get_valuations(bundle)
        total_valuation = 0
        for item in valuations:
            total_valuation += valuations[item]
        
        synergy_factor = 1.1 if len(bundle) > 1 else 1
        return total_valuation * synergy_factor

    def find_best_bundle(self, goods, max_bundle_size=10):
        best_value = 0
        best_bundle = []
        for size in range(1, max_bundle_size + 1):
            for bundle in combinations(goods, size):
                bundle_value = self.evaluate_bundle(bundle)
                if bundle_value > best_value:
                    best_value = bundle_value
                    best_bundle = bundle
        return best_bundle, best_value

    def regional_bidder_strategy(self):
        min_bids = self.get_min_bids()
        goods = self.get_goods()
        bids = {}

        best_bundle, best_bundle_value = self.find_best_bundle(goods)
        bundle_min_bid = sum(min_bids[good] for good in best_bundle)

        # Bidding strategy for identified bundle
        for good in goods:
            good_min_bid = min_bids[good]
            if good in best_bundle:
                bid_value = good_min_bid if good_min_bid < self.get_valuation(good) else self.get_valuation(good) 
                bids[good] = max(good_min_bid, bid_value)
            else:
                bids[good] = 0 

        return bids

    def get_bids(self):
        # if self.is_national_bidder():
        #     return self.national_bidder_strategy()
        # else:
        return self.regional_bidder_strategy()

    def update(self):
        pass

class Matt_Greedy_Agent(MyLSVMAgent):
    def setup(self):
        pass

    def get_averages(self, val_matrix):
        avg_matrix = np.zeros((3, 6))
        for iy, ix in np.ndindex(val_matrix.shape):
            neighbors = []
            if iy - 1 >= 0:
                neighbors.append(val_matrix[iy-1, ix])
            if iy + 1 < 3:
                neighbors.append(val_matrix[iy+1, ix])
            if ix - 1 >= 0:
                neighbors.append(val_matrix[iy, ix-1])
            if ix + 1 < 6:
                neighbors.append(val_matrix[iy, ix+1])
            avg_matrix[iy, ix] = np.mean(neighbors) if neighbors else 0
        return avg_matrix

    def evaluate_bundle(self, bundle):
        valuations = self.get_valuations(bundle)
        total_valuation = 0
        for item in valuations:
            total_valuation += valuations[item]
    
        return self.calc_total_utility(bundle)

    def find_best_bundle(self, goods, max_bundle_size=6):
        best_value = 0
        best_bundle = []
        for size in range(1, max_bundle_size + 1):
            for bundle in combinations(goods, size):
                bundle_value = self.evaluate_bundle(bundle)
                if bundle_value > best_value:
                    best_value = bundle_value
                    best_bundle = bundle
        return best_bundle, best_value

    def regional_bidder_strategy(self):
        min_bids = self.get_min_bids()
        goods = self.get_goods()
        bids = {}

        best_bundle, best_bundle_value = self.find_best_bundle(goods)
        bundle_min_bid = sum(min_bids[good] for good in best_bundle)

        # Bidding strategy for identified bundle
        for good in goods:
            good_min_bid = min_bids[good]
            if good in best_bundle:
                bid_value = good_min_bid if good_min_bid < self.get_valuation(good) else self.get_valuation(good) 
                bids[good] = max(good_min_bid, bid_value)
            else:
                bids[good] = 0  # Not bidding on non-bundle items

        return bids

    def get_bids(self):
        return self.regional_bidder_strategy()

    def update(self):
        pass

class Matt_Smart_Greedy_Agent(MyLSVMAgent):
    def setup(self):
        pass

    def find_best_bundles(self, goods, max_bundle_size=6, top_x_bundles = 10):
        best_bundles = []
        
        # for size in range(1, max_bundle_size + 1):
        for bundle in combinations(goods, max_bundle_size):
            total_valuation = self.calc_total_valuation(bundle)
            
            if len(best_bundles) != top_x_bundles:
                if best_bundles == []:
                    best_bundles.append((bundle, total_valuation))
                else:
                    for i in range(len(best_bundles)):
                        if total_valuation > best_bundles[i][1]:
                            best_bundles.insert(i,(bundle, total_valuation))
                            break
                                
            elif total_valuation > best_bundles[-1][1]:
                for i in range(len(best_bundles)):
                        if total_valuation > best_bundles[i][1]:
                            best_bundles.insert(i,(bundle, total_valuation))
                            best_bundles.pop(-1)
                            break                    
                                    
        return best_bundles

    def regional_bidder_strategy(self):
        bids = {}
        all_goods = self.get_goods()
        min_bids = self.get_min_bids()
        for good in min_bids:
            if  min_bids[good] < self.get_valuation(good):
                bids[good] = min_bids[good]
            else:
                bids[good] = 0

        best_bundles = self.find_best_bundles(all_goods)

        for bundle in best_bundles:
            utility = self.calc_total_utility(bundle[0])
            if utility > 0:
                for good in bundle[0]:
                    bids[good] = min_bids[good]
        return bids

    def get_bids(self):
        return self.regional_bidder_strategy()

    def update(self):
        pass

class MyAgent(MyLSVMAgent):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.model = PredictionNetwork()
        # if os.path.isfile(MODELS_PATH + self.name + ".pth"):
        if self.name != NAME:
            self.model.load_state_dict(load(MODELS_PATH + self.name+ ".pth"))

    def setup(self):
        pass
    
    def national_bidder_strategy(self):
        return self.regional_bidder_strategy()

    def regional_bidder_strategy(self):
        utility = self.get_valuation_as_array()
                 
        state = torch.unsqueeze(torch.unsqueeze(torch.Tensor(utility),0),0)
        with open(f"outputs/{self.name}.txt", "a") as f:
            if self.get_current_round() == 0:
                f.write("New Auction\n")
                f.write("\n")
                f.write("\n")
            np.savetxt(f, utility)

        action_probs, critic_value = self.model.forward(state)
        action_probs = torch.flatten(action_probs)
        bids = self.convert_to_bids(action_probs)
        return bids

    def convert_to_bids(self, action_values):        
        bids = {}
        for i, good in enumerate(self.get_goods()):
            if  self.get_min_bids()[good] <= self.get_valuation(good) + action_values[i]:
                bids[good] = self.get_min_bids()[good]
            else:
                bids[good] = 0
        return bids
    
    def update(self):
        pass 

    def get_bids(self):
        if self.is_national_bidder(): 
            return self.national_bidder_strategy()
        else: 
            return self.regional_bidder_strategy()

################### SUBMISSION #####################
my_agent_submission = Original_Training_Agent(NAME)
####################################################

def process_saved_game(filepath, data): 
    """ 
    Here is some example code to load in a saved game in the format of a json.gz and to work with it
    """
    print(f"Processing: {filepath}")
    
    # NOTE: Data is a dictionary mapping 
    with gzip.open(filepath, 'rt', encoding='UTF-8') as f:
        game_data = json.load(f)
        for agent, agent_data in game_data.items(): 
            if agent_data['valuations'] is not None: 
                # agent is the name of the agent whose data is being processed 
                agent = agent 
                
                # bid_history is the bidding history of the agent as a list of maps from good to bid
                bid_history = agent_data['bid_history']
                
                # price_history is the price history of the agent as a list of maps from good to price
                price_history = agent_data['price_history']
                
                # util_history is the history of the agent's previous utilities 
                util_history = agent_data['util_history']
                
                # util_history is the history of the previous tentative winners of all goods as a list of maps from good to winner
                winner_history = agent_data['winner_history']
                
                # elo is the agent's elo as a string
                elo = agent_data['elo']
                
                # is_national_bidder is a boolean indicating whether or not the agent is a national bidder in this game 
                is_national_bidder = agent_data['is_national_bidder']
                
                # valuations is the valuations the agent recieved for each good as a map from good to valuation
                valuations = agent_data['valuations']
                
                # regional_good is the regional good assigned to the agent 
                # This is None in the case that the bidder is a national bidder 
                regional_good = agent_data['regional_good']
            
                # TODO: If you are planning on learning from previously saved games enter your code below. 
                # won_items = (MyLSVMAgent.map_to_ndarray(my_agent_submission, winner_history[-1], np.dtype('U100')) == agent).reshape(1,18)
                # if len(price_history) == 0:
                #     continue
                
                values = np.array([MyLSVMAgent.map_to_ndarray(my_agent_submission, valuations)])
                # prices = np.array(list(map(lambda x: MyLSVMAgent.map_to_ndarray(my_agent_submission, x), price_history)))

                if agent not in data:
                    data[agent] = [values]
                else:
                    # if len(elo) != 0:
                    # data[agent] = np.append(data[agent], values-prices, axis=0)
                    data[agent].append(values)
                
                # prices = np.array(list(map(lambda x: MyLSVMAgent.map_to_ndarray(my_agent_submission, x).flatten(), price_history)))
                # values = np.array([MyLSVMAgent.map_to_ndarray(my_agent_submission, valuations).flatten()])
                # if datax is None:
                #     datax = values - prices
                # else:
                #     datax = np.append(datax,values - prices, axis=0)
        return data
        
def process_saved_dir(dirpath): 
    """ 
     Here is some example code to load in all saved game in the format of a json.gz in a directory and to work with it
    """
    data = {}
    for filename in os.listdir(dirpath):
        if filename.endswith('.json.gz'):
            filepath = os.path.join(dirpath, filename)
            data = process_saved_game(filepath, data)
    return data
            

if __name__ == "__main__":
    
    ### DO NOT TOUCH THIS #####
    # agent = MyAgent(NAME)
    # arena = LSVMArena(
    #     num_cycles_per_player = 3,
    #     timeout=1,
    #     local_save_path="saved_games",
    #     players=[
    #         agent,
    #         MyAgent("CP - MyAgent"),
    #         MyAgent("CP2 - MyAgent"),
    #         MyAgent("CP3 - MyAgent"),
    #         MinBidAgent("Min Bidder"), 
    #         JumpBidder("Jump Bidder"), 
    #         TruthfulBidder("Truthful Bidder"), 
    #     ]
    # )

    arena = LSVMArena(
        num_cycles_per_player = 3,
        timeout=1,
        local_save_path="saved_games",
        players=[
            MinBidAgent("Min Bidder"),
            MinBidAgent("Min Bidder2"),
            JumpBidder("Jump Bidder"), 
            JumpBidder("Jump Bidder2"), 
            TruthfulBidder("Truthful Bidder"),  
            Original_Training_Agent("Original_Training_Agent")
        ]
    )
    
    start = time.time()
    with open('Original_Training_Agent.txt', 'w') as sys.stdout:
        arena.run()
    sys.stdout = sys.__stdout__
    end = time.time()
    print(f"{end - start} Seconds Elapsed")
