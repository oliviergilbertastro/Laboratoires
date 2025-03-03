import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

COLORS = ["blue", "red", "green", "orange"]



def loadFile(source="Am", detector="XR100T_CdTe", iteration=0, res=256):
    time_exposure[detectors.index(detector)] += time_exposures_list[detectors.index(detector)][iteration]
    return np.loadtxt(f"PHY-3004/SPEC_X/{source}_{res}_{detector}_{iteration}.mca", skiprows=12, max_rows=res, dtype=float)

detectors = ["XR100T_CdTe", "XR100CR_Si"]
time_exposures_list = [[60.971000,61.089000,61.151000,63.304000,70.676000],[60.180000,62.582000,108.369000,61.253000,60.936000]] # temps total d'exposition pour chaque détecteur

FWHM_save = {"XR100T_CdTe":[[],[],[]], "XR100CR_Si":[[],[],[]]}
Integrals_save = {"XR100T_CdTe":[[],[],[]], "XR100CR_Si":[[],[],[]]}

for iteration in range(5):
    data = []
    time_exposure = [0,0]
    for detector in detectors:
        data.append(loadFile(res=1024, detector=detector, iteration=iteration))
        if False:
            for i in range(1,5):
                data[-1]+=(loadFile(res=1024, detector=detector, iteration=i))

    data[0] = data[0]/time_exposure[0] # count/s
    data[1] = data[1]/time_exposure[1] # count/s

    for n in range(len(detectors)):
        plt.plot(data[n], label=detectors[n])
    plt.legend()
    plt.xlabel("Channel")
    plt.ylabel("Counts/s")
    plt.show()

    # Étalonnage de professionnel:
    from scipy.optimize import curve_fit
    x_data = []
    etalonnages = []
    for i in range(len(detectors)):
        x_data.append(np.array(range(1024)))
        peaks_indices = [[163,208,692],[184,234,785]][i]
        etalonnages.append(curve_fit(lambda x,a,b: x*a+b, peaks_indices, [13.95,17.74,59.54])[0])
        x_data[i] = x_data[i]*etalonnages[i][0]+etalonnages[i][1]

    for n in range(len(detectors)):
        plt.plot(x_data[n], data[n], label=detectors[n])
    plt.legend()
    plt.xlabel("keV")
    plt.ylabel("Count")
    plt.show()

    def gaussian(x, sig, b, mu, c=0):
        return (
            b / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x-mu) / sig, 2.0) / 2) + c
        )

    print("Temps exposition:", time_exposure)


    # Integrate the gaussians to get the total area
    from scipy.integrate import quad

    FWHMs = []
    for m, pics in enumerate([(155,170),(202,215),(628,707)]):

        gaussiennes = []
        for n in range(len(detectors)):
            new_pics = (int(pics[0]*etalonnages[0][0]/etalonnages[n][0]), int(pics[1]*etalonnages[0][0]/etalonnages[n][0]))
            gaussiennes.append(data[n][new_pics[0]:new_pics[1]])


        res = []
        FWHMs.append([])
        print(f"{[13.95,17.74,59.54][m]}keV:")
        for n in range(len(detectors)):
            res.append(curve_fit(gaussian, range(len(gaussiennes[n])), gaussiennes[n], p0=[len(gaussiennes[n]), np.max(gaussiennes[n])*(np.sqrt(2.0 * np.pi) * 10), len(gaussiennes[n])/2, np.min(gaussiennes[n])], bounds=[(0,0,0,0),(np.inf,np.inf,np.inf,np.inf)])[0])
            #print(res)
            FWHMs[m].append(2*np.sqrt(2*np.log(2))*res[n][0])
            print(f"FWHM {detectors[n]} = {FWHMs[m][n]*etalonnages[n][0]*1000} eV")
            integral = quad(gaussian, a=-np.inf, b=np.inf, args=(res[-1][0], res[-1][1], res[-1][2], 0))[0]
            print(f"counts/s {detectors[n]} = {integral}")
            FWHM_save[detectors[n]][m].append(FWHMs[m][n])
            Integrals_save[detectors[n]][m].append(integral)

        x_sim = np.linspace(0, len(gaussiennes[0]), 1000)
        for n in range(len(detectors)):
            plt.bar(range(len(gaussiennes[n])), gaussiennes[n], width=1, color=COLORS[n], alpha=0.4, label=detectors[n])
            plt.plot(x_sim, gaussian(x_sim, res[n][0], res[n][1], res[n][2], res[n][3]), color=COLORS[n])
        plt.legend()
        plt.xlabel("Channel")
        plt.ylabel("Count")
        plt.show()

infos = {"FWHMs": FWHM_save, "Count_rates": Integrals_save}
with open("PHY-3004/SPEC_X/infos.json", "w") as outfile: 
    json.dump(infos, outfile, indent=4)