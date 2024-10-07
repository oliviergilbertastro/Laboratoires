import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from matplotlib.image import imread

from scipy.signal import boxcar
from pylab import r_

def smooth(x, smoothing_param=3):
    window_len=smoothing_param*2+1
    s=r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    w=boxcar(smoothing_param*2+1)
    y=np.convolve(w/np.sum(w),s,mode='valid')
    return y[smoothing_param:-smoothing_param] 


img_folder = "PHY-3002/Photos_RIP/"


sample_img = np.array(imread(f"{img_folder}sample.jpg"))[:,:,:]
plt.imshow(sample_img, origin="lower")
plt.show()


MM_PER_PX = 10/(92)

freq = 241

img = np.array(imread(f"{img_folder}{freq}_Hz.jpg"))[:,:,0]
#m=0: 1783, 1032

img_y_axis_pixels = np.linspace(0, img.shape[0], img.shape[0])
# convert to mm from m=0:
img_y_axis_pixels -= 1032
img_y_axis_pixels *= MM_PER_PX


plt.imshow(img, origin="lower", cmap="Reds")
plt.show()

img_profile = np.mean(img, axis=1)/np.max(np.mean(img, axis=1))
plt.plot(img_profile)
plt.show()
img = img[:,1770:1820]
img_profile = np.mean(img, axis=1)/np.max(np.mean(img, axis=1))
img_profile = smooth(img_profile, smoothing_param=10)
plt.plot(img_y_axis_pixels, img_profile)
plt.show()