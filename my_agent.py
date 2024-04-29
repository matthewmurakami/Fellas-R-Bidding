from agt_server.agents.base_agents.lsvm_agent import MyLSVMAgent
from agt_server.local_games.lsvm_arena import LSVMArena
from agt_server.agents.test_agents.lsvm.min_bidder.my_agent import MinBidAgent
from agt_server.agents.test_agents.lsvm.jump_bidder.jump_bidder import JumpBidder
from agt_server.agents.test_agents.lsvm.truthful_bidder.my_agent import TruthfulBidder
import time
import os
import random
import gzip
import json
from path_utils import path_from_local_root
import numpy as np




KERNEL = np.array([[1/9, 1/9, 1/9],
                   [1/9, 1/9, 1/9],
                   [1/9, 1/9, 1/9]])


NAME = 'Fellas-R-Bidding'

class MyAgent(MyLSVMAgent):
    def setup(self):
        #TODO: Fill out with anything you want to initialize each auction
        pass 
    
    def national_bidder_strategy(self): 
        return self.regional_bidder_strategy()


    def regional_bidder_strategy(self):
        # TODO: Fill out with your regional bidder strategy
        min_bids = self.get_min_bids()
        valuations = self.get_valuations() 
        goods = self.get_goods()

        bids = {} 

        for good in goods:

            self.proximity()




            good_val = valuations[good]
            good_min_bid = min_bids[good]

            if good_min_bid <= good_val:
                bids[good] = good_min_bid
            else:
                bids[good] = None

        return bids


    def get_bids(self):
        if self.is_national_bidder(): 
            return self.national_bidder_strategy()
        else: 
            return self.regional_bidder_strategy()
    



################### SUBMISSION #####################
my_agent_submission = MyAgent(NAME)
####################################################


def process_saved_game(filepath): 
    """ 
    Here is some example code to load in a saved game in the format of a json.gz and to work with it
    """
    print(f"Processing: {filepath}")
    
    # NOTE: Data is a dictionary mapping 
    with gzip.open(filepath, 'rt', encoding='UTF-8') as f:
        game_data = json.load(f)
        datax = None
        datay = np.array([])
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
                datay = np.append(datay,np.array(util_history))

                prices = np.array(list(map(lambda x: MyLSVMAgent.map_to_ndarray(my_agent_submission, x).flatten(), price_history)))
                values = np.array([MyLSVMAgent.map_to_ndarray(my_agent_submission, valuations).flatten()])
                if datax is None:
                    datax = values - prices
                else:
                    datax = np.append(datax,values - prices, axis=0)
        return datax, datay
        
def process_saved_dir(dirpath): 
    """ 
     Here is some example code to load in all saved game in the format of a json.gz in a directory and to work with it
    """
    datax = None
    datay = np.array([])
    for filename in os.listdir(dirpath):
        if filename.endswith('.json.gz'):
            filepath = os.path.join(dirpath, filename)
            filedatax, filedatay = process_saved_game(filepath)
            if datax is None:
                datax = filedatax
            else:
                datax = np.append(datax, filedatax,axis=0)
            datay = np.append(datay, filedatay)
    return datax, datay
            

if __name__ == "__main__":
    
    # Heres an example of how to process a singular file 
    # process_saved_game(path_from_local_root("saved_games/2024-04-08_17-36-34.json.gz"))
    # or every file in a directory 
    # process_saved_dir(path_from_local_root("saved_games"))
    
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
            TruthfulBidder("Truthful Bidder2") 
        ]
    )
    
    start = time.time()
    arena.run()
    end = time.time()
    print(f"{end - start} Seconds Elapsed")
