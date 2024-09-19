import numpy as np
from tqdm import tqdm

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
    num7(N_rolls=1000000)