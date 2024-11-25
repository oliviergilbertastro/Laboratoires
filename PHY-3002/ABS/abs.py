import numpy as np
import matplotlib.pyplot as plt

from matplotlib.image import imread




img1 = np.array(imread(f"PHY-3002/ABS/photos/IMG_1720.jpg"))
img2 = np.array(imread(f"PHY-3002/ABS/photos/IMG_1726.jpg"))

ax1, ax2 = plt.subplot(121), plt.subplot(122)
ax1.imshow(img1)
ax2.imshow(img2)
plt.show()


# Crop the images so it's only the middle 200 pixel in height
img1 = img1[1100:1300,:,:]
img2 = img2[1100:1300,:,:]
ax1, ax2 = plt.subplot(211), plt.subplot(212)
ax1.imshow(img1)
ax2.imshow(img2)
plt.show()

#horizontal pixel position of Hg 546.074nm line on img1: 863
#horizontal pixel position of Hg 546.074nm line on img2: 1838

# get the intensity profiles:
img1 = np.mean(img1, axis=2) #grayscale
profile1 = np.median(img1, axis=0)
img2 = np.mean(img2, axis=2) #grayscale
profile2 = np.median(img2, axis=0)
ax1, ax2 = plt.subplot(211), plt.subplot(212)
ax1.plot(profile1)
ax2.plot(profile2)
ax1.set_xlabel("pixel", fontsize=15)
ax1.set_ylabel("intensité", fontsize=15)
ax2.set_xlabel("pixel", fontsize=15)
ax2.set_ylabel("intensité", fontsize=15)
plt.show()

Hg_line_normalization = 145/47

#let's combine both profiles to get one full spectrum
profile = np.concatenate([profile2[:1838]/(Hg_line_normalization),profile1[863:]])
profile = profile[::-1]
ax1 = plt.subplot(111)
plt.plot(profile)
ax1.set_xlabel("pixel", fontsize=15)
ax1.set_ylabel("intensité", fontsize=15)
plt.show()


lines_nm = np.array([508.582,546.074,576.960,579.066])
lines_px = np.array([1070,1952,2670,2719])

from scipy.optimize import curve_fit
def convert_px_to_lambda(px, a, b):
    return px*a+b
res = curve_fit(convert_px_to_lambda, lines_px, lines_nm)[0]
print(res[0])
ax1 = plt.subplot(111)
plt.plot(lines_px, lines_nm, "o")
plt.plot(lines_px, convert_px_to_lambda(lines_px, res[0], res[1]), "--")
ax1.set_xlabel("Pixel", fontsize=15)
ax1.set_ylabel("$\lambda$ [nm]", fontsize=15)
plt.show()




