import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

def smooth(a, smoothing=3):
    """
    smoothing : int
    """
    new_a = []
    for i in range(smoothing):
        new_a.append(np.mean(np.array(a)[0:i+smoothing]))
    for i in range(smoothing, len(a)-smoothing):
        new_a.append(np.mean(np.array(a)[i-smoothing:i+smoothing]))
    for i in range(smoothing):
        new_a.append(np.mean(np.array(a)[i-smoothing:]))
    return np.array(new_a)



img = np.array(imread(f"PHY-3002/Photos_RIP/IMG_2714.JPG"))[:,:,:]
img = img[:,1600:2000,0]
plt.imshow(img)
plt.show()

profile = np.mean(img, axis=1)
plt.plot(profile)
plt.show()

smooth_profile = smooth(profile, smoothing=5)
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)
ax1.plot(profile)
ax2.plot(smooth_profile)
plt.show()

from scipy.signal import find_peaks

peaks_idx = find_peaks(smooth_profile, distance=50, height=2.5)[0]

print(peaks_idx)

plt.plot(profile, color="black", label="data")
plt.plot(smooth_profile, color="red", label="smooth")
plt.plot(peaks_idx, smooth_profile[peaks_idx], "o", color="blue", label="peaks")
plt.legend()
plt.show()