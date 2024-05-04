import re
import os


def proccess_their_output(file_path):
    output = {}
    elo_section = False
    with open(file_path) as file:
        for line in file:
            line = line.strip()
            if not elo_section:

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
           
    return output


def proccess_our_output():
    for filename in os.listdir("outputs/"):
        with open(f'outputs/{filename}', "r") as f:
            count = 0
            for line in f:
                line = line.strip()
                if line == "New Auction":
                    count+=1
            print("items: ", count)
            
            f.close()
                
