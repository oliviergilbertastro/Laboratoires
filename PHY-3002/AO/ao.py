import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from matplotlib.image import imread

from scipy.signal import boxcar, find_peaks


def get_peaks_indices(valeurs: np.array,
                                      hauteur_minimum: int = None,
                                      distance_minumum: int = None):
    """
    Cette méthode utilise la méthode find_peaks de scipy pour trouver les indices de l'array correspondant aux maximums.

    :param valeurs: array contenant toutes les valeurs
    :param colonne_y: index de la colonne contenant les valeurs dépendantes
    :param hauteur_minimum: Optionnel! Celui-ci indique la hauteur minimale entre
                            deux pics pour guider l'algorithme de scipy
                            Cette valeur devrait être inférieure à la hauteur du plus grand pic.
                            de l'array numpy (<array.max())
    :param distance_minumum:Optionnel! Celui-ci indique la distance minimale entre
                            deux pics pour guider l'algorithme de scipy
                            Cette valeur devrait être inférieure à la longueur totale
                            de l'array numpy divisée par le nombre total de pics (<(array.shape[1]/nb de pics))

    :return: liste des indices correspondant aux emplacements des maximums
    """
    peaks, _ = find_peaks(valeurs, height=hauteur_minimum, distance=distance_minumum)
    return peaks

from pylab import r_

from scipy.ndimage import rotate

def smooth(x, smoothing_param=3):
    window_len=smoothing_param*2+1
    s=r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    w=boxcar(smoothing_param*2+1)
    y=np.convolve(w/np.sum(w),s,mode='valid')
    return y[smoothing_param:-smoothing_param] 

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

CM_PER_PX = 1/(243.347)

frequencies = [34,36,38,40,42,44,46] # MHz
frequencies = [42,44,46] # MHz

# y,x
centers = [[428,1041], [669,1131], [725,1102]] # bleu, vert, rouge
#ranges = [[350,500], []]
ranges = [[centers[0][0]-100,centers[0][0]+100],[centers[1][0]-100,centers[1][0]+100],[centers[2][0]-100,centers[2][0]+100]]

choice = input("Choose color [bleu,vert,rouge]\n")
if choice == "":
    choice = ["bleu","vert","rouge"]
else:
    choice = [choice]



if_plot = True


for couleur in choice:
    i = ["bleu","vert","rouge"].index(couleur)
    img_folder = "PHY-3002/AO/photos/"

    
    sample_img = np.array(imread(f"{img_folder}{couleur}/34.jpg"))[:,:,:]
    if if_plot:
        plt.imshow(sample_img, origin="lower")
        plt.show()

    for f in frequencies:
        img = np.array(imread(f"{img_folder}{couleur}/{f}.jpg"))[:,:,2-i]
        
        #m=0: 1783, 1032

        img_x_axis_pixels = np.linspace(0, img.shape[1], img.shape[1])
        # convert to mm from m=0:
        img_x_axis_pixels -= centers[i][1]
        img_x_axis_pixels *= CM_PER_PX

        img = img[ranges[i][0]:ranges[i][1], centers[i][1]-500:centers[i][1]+500]
        img_x_axis_pixels = img_x_axis_pixels[centers[i][1]-500:centers[i][1]+500]
        if if_plot:
            plt.imshow(img, origin="lower", cmap=["Blues","Greens","Reds"][i])
            plt.title(f"{couleur} {f}MHz", fontsize=15)
            plt.show()

        img_profile = np.mean(img, axis=0)/np.max(np.mean(img, axis=0))
        if if_plot:
            plt.plot(img_profile)
            plt.show()
        
        ok_bad = False
        ok_good = False
        img_profile = np.mean(img, axis=0)/np.max(np.mean(img, axis=0))
        img_profile = smooth(img_profile, smoothing_param=3)
        peaks = list(get_peaks_indices(img_profile, distance_minumum=70))
        while not (ok_bad and ok_good):
            ok_good = False
            for k in range(len(img_x_axis_pixels[peaks])):
                print(f"Pic {k}: {img_x_axis_pixels[peaks][k]}")

            plt.plot(img_x_axis_pixels[peaks], img_profile[peaks], "o")
            plt.plot(img_x_axis_pixels, img_profile)
            plt.show()

            bad_peaks = input("Entrez l'indice de chaque pic indésirable séparés d'une virgule, puis appuyez sur 'ENTER':\n")
            if bad_peaks == "":
                ok_bad = True
                break
            bad_peaks = bad_peaks.split(sep=",")
            bad_peaks.sort()
            print(bad_peaks)
            for k in bad_peaks[::-1]:
                peaks.pop(int(k))
            while not ok_good:
                good_peaks = input("Entrez la valeur en cm du pic à rajouter, puis appuyer sur 'ENTER':\n")
                if good_peaks == "":
                    ok_good = True
                    break
                peaks.append(find_nearest_index(img_x_axis_pixels,float(good_peaks)))
            peaks.sort()
        array_to_save = np.array([img_x_axis_pixels[peaks], np.array(range(len(peaks)))-int(len(peaks)/2)])
        np.savetxt(f"PHY-3002/AO/data/{couleur}_{f}.txt", array_to_save)