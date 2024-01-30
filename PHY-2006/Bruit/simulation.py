import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import gauss

class SignalBruit():
    def __init__(self, tension, period, noise_mean, noise_std, number_of_cycles=10):
        self.tension = tension
        self.period = period
        self.data = []
        for n in range(number_of_cycles):
            for i in range(period):
                if i > period/2:
                    self.data.append(self.tension+gauss(noise_mean, noise_std))
                else:
                    self.data.append(gauss(noise_mean, noise_std))
    
    def show(self):
        plt.plot(self.data)
        plt.xlabel(r'Temps (ms)', size=17)
        plt.ylabel(r'Tension (V)', size=17)
        plt.show()

    def saveTo(self, filename):
        np.savetxt(filename, self.data)

signal = SignalBruit(
                        tension=5,
                        period=100,
                        noise_mean=0,
                        noise_std=0.05,
                        number_of_cycles=50
                        )
signal.saveTo('PHY-2006/Bruit/signalSimu.txt')