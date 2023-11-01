import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_values_from_file(filename):
    values = pd.read_csv(filename, delimiter="\t", decimal=",", skiprows=21)
    return values




def create__osci_plot(filename, name):
    values = pd.read_csv(filename, delimiter=",", decimal=".", skiprows=1)
    x = np.array(values.iloc[:, 0])
    y = np.array(values.iloc[:, 1])
    z = np.array(values.iloc[:, 2])

    
    
    ax1 = plt.subplot(111)
    ticklabels = ax1.get_xticklabels()
    ticklabels.extend( ax1.get_yticklabels() )
    for label in ticklabels:
        label.set_fontsize(10)
    plt.plot(x, y, label="Canal 1")
    plt.plot(x, z, label="Canal 2")
    plt.legend()
    plt.suptitle(f'Oscilloscope avec impulsion de durée {name}', size=17)
    plt.ylabel(r'Tension [V]', size=17)
    plt.xlabel(r'Temps [s]', size=17)
    plt.savefig(r'C:\Users\olivi\Desktop\Devoirs\PhysElectronique\figures\lab5'+f"\oscilloscope_{name}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    return True


#create__osci_plot("lab5/usb/scope_h2.csv", "")
#create__osci_plot("lab5/usb/scope_h3.csv", "")

create__osci_plot("lab5/usb/scope_i5.csv", "5W")
create__osci_plot("lab5/usb/scope_i6.csv", "W")
create__osci_plot("lab5/usb/scope_i7.csv", "W (court-circuité)")
create__osci_plot("lab5/usb/scope_i8.csv", "5W (court-circuité)")