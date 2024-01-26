import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tqdm import tqdm

src_img = imread('PHY-2006/holes.jpg')

mean_img = np.mean(src_img, 2)

def net_flux(image, target_pos, noise_pos, radius=40, noise_radius=50, if_plot=False, verbose=False):
    '''
    Calculates the approximate flux of a source by substracting the noise of its nearby background

    Parameters:
    image: image data with the source (2D array)
    target_pos: list of position of the source (e.g. [100, 200])
    radius: approximate radius of the source object
    if_plot: boolean, if True, plot only the flux within the radius
    
    Returns the estimated flux of the source
    '''
    x0, y0 = target_pos[0], target_pos[1]
    x1, y1 = noise_pos[0], noise_pos[1]
    pixels_inside, pixels_noise = 0, 0
    within_flux = np.zeros(image[y0-2*radius:y0+2*radius, x0-2*radius:x0+2*radius].shape)
    noise_flux = np.zeros(image[y1-2*noise_radius:y1+2*noise_radius, x1-2*noise_radius:x1+2*noise_radius].shape)
    for y, line in enumerate(image[y0-2*radius:y0+2*radius, x0-2*radius:x0+2*radius]):
        for x, val in enumerate(line):
            if np.sqrt((x-2*radius-1)**2+(y-2*radius)**2)<=radius:
                within_flux[y, x] = val
                pixels_inside += 1
    for y, line in enumerate(image[y1-2*noise_radius:y1+2*noise_radius, x1-2*noise_radius:x1+2*noise_radius]):
        for x, val in enumerate(line):
            if np.sqrt((x-2*noise_radius-1)**2+(y-2*noise_radius)**2)<=noise_radius:
                noise_flux[y, x] = val
                pixels_noise += 1
    
    

    flux_within = np.sum(within_flux)
    flux_noise = np.sum(noise_flux)
    flux_source = flux_within/pixels_inside# - (flux_noise/pixels_noise)*pixels_inside

    return flux_source


radius = int(input("Radius?"))
radial_intensity = []
surface_profile = []
for i in tqdm(range(radius)):
    radial_intensity.append(net_flux(mean_img, [895,422], [0,0], i, noise_radius=0))
    if i == 0:
        surface_profile.append(radial_intensity[i])
    else:
        surface_profile.append(radial_intensity[i]-radial_intensity[i-1])



plt.imshow(mean_img)
plt.show()

ax1 = plt.subplot(111)
ticklabels = ax1.get_xticklabels()
ticklabels.extend( ax1.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(14)
plt.plot(range(radius), surface_profile, color="red", label="données")
plt.xlabel(r'Distance (pixels)', size=17)
plt.ylabel(r'Intensité ($I_0$)', size=17)
plt.show()