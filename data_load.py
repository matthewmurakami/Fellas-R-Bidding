import re
import os
import glob
import numpy as np

def process_their_output(file_path):
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
                        output[utility.group(1)] = [0,[float(utility.group(2))]]
                    else:
                        output[utility.group(1)][1].append(float(utility.group(2)))
                    continue
                if re.match("Agent Name  Final Score   ELO", line):
                    elo_section = True
                    continue
            else:
                elo = re.match("^\d+[\s]+([^\s]+)[\s]+(-?\d*[.,]?\d*)[\s]+(\d+)$", line)

                if elo:
                    output[elo.group(1)][0] = float(elo.group(2))
                    
    return output

def process_our_output():

    utility_history = {}

    for filename in os.listdir("outputs/"): 
           
        first_iter = True
        auction_list = np.array([])
        round_list = []
        name = filename.split(".")[0]
        utility_history[name] = []

        file = open(f'outputs/{filename}')
        while(True):
            line1 = file.readline().strip()
            

            if line1 == "":
                round_list = np.asarray(round_list)
                auction_list = round_list
                utility_history[name].append(auction_list)
                break

            line2 = file.readline().strip()
            line3 = file.readline().strip()

            if line1 == "New Auction":
                if first_iter:
                    first_iter = False
                    auction_list = np.array([])
                    continue

                round_list = np.asarray(round_list)
            
                if auction_list.size == 0:
                    auction_list = round_list
                else:
                    auction_list = np.concatenate((auction_list,round_list),axis=0)

                utility_history[name].append(auction_list)
                auction_list = np.array([])
                round_list = []
                continue
            else:
                arr1 = line1.split(" ")
                arr2 = line2.split(" ")
                arr3 = line3.split(" ")
                np_arr = np.asarray([arr1,
                                     arr2,
                                     arr3], dtype=np.float32)
                round_list.append(np_arr)
                    
        file.close()

    return utility_history 

# def process_our_output():
#     utility_history = {}
#     for filename in os.listdir("outputs/"):
#         first_iter = True
        
#         auction_list = np.array([])

#         round_list = []
#         utility_history[filename] = []


#         file = open(f'outputs/{filename}')
#         while(True):
#             line1 = file.readline().strip()

#             if line1 == "":
#                 break

#             line2 = file.readline().strip()
#             line3 = file.readline().strip()

#             if line1 == "New Auction":
#                 if first_iter:
#                     first_iter = False
#                     auction_list = np.array([])
#                     continue

#                 round_list = np.asarray(round_list)
            
#                 if auction_list.size == 0:
#                     print("TRUE")
#                     auction_list = round_list
#                 else:
#                     print("auction_list:",auction_list.shape)
#                     auction_list = np.concatenate((auction_list,round_list),axis=0)

#                 auction_list = np.array([])
#                 round_list = []
#                 continue
#             else:
#                 arr1 = line1.split(" ")
#                 arr2 = line2.split(" ")
#                 arr3 = line3.split(" ")
#                 np_arr = np.asarray([arr1,
#                                      arr2,
#                                      arr3])
#                 round_list.append(np_arr)

#         print("EOF auction_list:", auction_list.shape)
    
#         utility_history[filename].append(auction_list)
#         file.close()

#     return utility_history
