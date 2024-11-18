import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from matplotlib.image import imread

from scipy.signal import boxcar
from pylab import r_

from scipy.ndimage import rotate

def smooth(x, smoothing_param=3):
    window_len=smoothing_param*2+1
    s=r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    w=boxcar(smoothing_param*2+1)
    y=np.convolve(w/np.sum(w),s,mode='valid')
    return y[smoothing_param:-smoothing_param] 

CM_PER_PX = 1/(243.347)

frequencies = [34,36,38,40,42,44,46] # MHz

# y,x
centers = [[428,1041], [669,1131], [725,1102]] # bleu, vert, rouge
#ranges = [[350,500], []]
ranges = [[centers[0][0]-100,centers[0][0]+100],[centers[1][0]-100,centers[1][0]+100],[centers[2][0]-100,centers[2][0]+100]]

choice = input("Choose color [bleu,vert,rouge]\n")
if choice == "":
    choice = ["bleu","vert","rouge"]
else:
    choice = [choice]
for couleur in choice:
    i = ["bleu","vert","rouge"].index(couleur)
    img_folder = "PHY-3002/AO/photos/"

    
    sample_img = np.array(imread(f"{img_folder}{couleur}/34.jpg"))[:,:,:]
    plt.imshow(sample_img, origin="lower")
    plt.show()

    for f in frequencies:
        img = np.array(imread(f"{img_folder}{couleur}/{f}.jpg"))[:,:,2-i]
        
        #m=0: 1783, 1032

        img_x_axis_pixels = np.linspace(0, img.shape[1], img.shape[1])
        # convert to mm from m=0:
        img_x_axis_pixels -= centers[i][1]
        img_x_axis_pixels *= CM_PER_PX

        img = img[ranges[i][0]:ranges[i][1], 250:1750]
        img_x_axis_pixels = img_x_axis_pixels[250:1750]

        plt.imshow(img, origin="lower", cmap=["Blues","Greens","Reds"][i])
        plt.title(f"{couleur} {f}MHz", fontsize=15)
        plt.show()

        img_profile = np.mean(img, axis=0)/np.max(np.mean(img, axis=0))
        plt.plot(img_profile)
        plt.show()
        img_profile = np.mean(img, axis=0)/np.max(np.mean(img, axis=0))
        img_profile = smooth(img_profile, smoothing_param=10)
        plt.plot(img_x_axis_pixels, img_profile)
        plt.show()