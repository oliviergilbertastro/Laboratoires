import numpy as np
from tqdm import tqdm
import os
import sys
import parse
#code_cours = os.path.dirname(os.path.realpath(__file__))[-4:]
#parent_dir = parse.parse("{}\PHY-"+code_cours, os.path.dirname(os.path.realpath(__file__)))[0]
#sys.path.append(parent_dir)

sys.path.append(parse.parse("{}\PHY-"+os.path.dirname(os.path.realpath(__file__))[-4:], os.path.dirname(os.path.realpath(__file__)))[0])
from utils import *

def num1():
    prob = 1
    for i in range(1,11):
        prob *= (365-i)/365 # prob de ne pas partager la fête avec quelqu'un
    print(prob)
    prob = 1-prob
    print(prob)
    pass

def num7(N_rolls=1000, nb_of_dice=3, if_plot=True):
    totals = []
    for i in tqdm(range(N_rolls)):
        tot_dots_of_roll = 0
        for k in range(nb_of_dice):
            tot_dots_of_roll += np.random.randint(1,7)
        totals.append(tot_dots_of_roll)

    totals = np.array(totals)
    totals_each = []
    probs = []
    for i in range(18):
        totals_each.append(totals[totals==i+1])
        probs.append(len(totals_each[-1])/N_rolls)
        print(f"Probability of rolling {i+1} dots is around {probs[-1]}")

    return totals

if __name__ == "__main__":
    print_color("****************************Numéro D1.1**********************************")
    num1()
    print_color("****************************Numéro D1.7**********************************")
    num7(N_rolls=1000000)