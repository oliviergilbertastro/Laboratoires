import numpy as np
import matplotlib.pyplot as plt
import mpl_interactions.ipyplot as iplt
from matplotlib.widgets import Slider

data = np.loadtxt(f"PHY-3002/ABS/olivier_xavier.csv", delimiter=";", skiprows=104, comments="[")

data = data.T

wav = data[0,:]
intensity = data[1,:]
if False:
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, sharex=ax1)
    ax1.plot(wav)
    ax2.plot(intensity)
    plt.show()

bounds = [18180,26715]
wav = wav[bounds[0]:bounds[1]]
intensity = intensity[bounds[0]:bounds[1]]
intensity = intensity/np.max(intensity)


# Find the absorption lines
from scipy.signal import boxcar, find_peaks
def get_absorption_lines_indices(valeurs: np.array,
                                      hauteur_minimum: int = None,
                                      distance_minimum: int = None):
    peaks, _ = find_peaks(-valeurs, height=hauteur_minimum, distance=distance_minimum)
    return peaks


from scipy.signal import boxcar
from pylab import r_
def smooth(x, smoothing_param=3):
    window_len=smoothing_param*2+1
    s=r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    w=boxcar(smoothing_param*2+1)
    y=np.convolve(w/np.sum(w),s,mode='valid')
    return y[smoothing_param:-smoothing_param] 

def abs_lines_x(sens, inten):
    abs_lines = get_absorption_lines_indices(inten, distance_minimum=200/sens)
    return wav[abs_lines]
def abs_lines_y(sens, inten):
    abs_lines = get_absorption_lines_indices(inten, distance_minimum=200/sens)
    return inten[abs_lines]
fig, ax = plt.subplots()
fig.set_size_inches(6, 7.25)
plt.subplots_adjust(bottom=0.25)
ax1 = plt.axes([0.25, 0.1, 0.65, 0.03])
ax2 = plt.axes([0.25, 0.15, 0.65, 0.03])
slider1 = Slider(ax1, label="Sensitivity", valmin=1, valmax=10, valinit=1)
slider2 = Slider(ax2, label="Smoothing", valmin=1, valmax=10, valinit=1, valstep=int(1))
# Plot the base spectrum
spectrum_line, = ax.plot(wav, intensity, color="black", label="Spectrum")
#controls = iplt.imshow(controlVisual, yes=check1, ax=ax, origin="lower")
sc = ax.scatter(abs_lines_x(1, intensity), abs_lines_y(1, intensity), color="red", label="Absorption lines")

# Update function
def update1(val):
    sens = slider1.val
    smoothed_intensity = smooth(intensity, smoothing_param=slider2.val)
    #smoothed_intensity = smoothed_intensity-np.min(smoothed_intensity)
    #smoothed_intensity = smoothed_intensity/np.max(smoothed_intensity)
    sc.set_offsets(np.column_stack((abs_lines_x(sens, smoothed_intensity), abs_lines_y(sens, smoothed_intensity))))
    fig.canvas.draw_idle()
def update2(val):
    smoothing = slider2.val
    smoothed_intensity = smooth(intensity, smoothing_param=smoothing)
    #smoothed_intensity = smoothed_intensity-np.min(smoothed_intensity)
    #smoothed_intensity = smoothed_intensity/np.max(smoothed_intensity)
    spectrum_line.set_ydata(smoothed_intensity)
    update1(slider1.val)
    fig.canvas.draw_idle()

# Connect slider to update function
slider1.on_changed(update1)
slider2.on_changed(update2)
plt.xlabel("$\lambda$ [nm]", fontsize=15)
plt.ylabel("Intensité normalisée", fontsize=15)
plt.show()


abs_waves = abs_lines_x(slider1.val, smooth(intensity, smoothing_param=slider2.val))
theoretical_lines = np.array([508.674,509.529,510.433,511.388,512.396,513.458,514.576,515.751,516.984,518.278,519.634,521.052,522.535,524.083,525.699,527.383,529.137,530.962,532.860,534.831,536.877,538.999,541.199,543.478,545.838,548.279,547.520,550.803,549.853,553.411,552.268,556.106,554.767,558.888,557.351,561.759,560.022,564.721,562.781,567.775,565.631,570.924,568.572,574.169,571.606,574.736,577.512,577.952])

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

compared_lines = []
for i in range(len(abs_waves)):
    idx = find_nearest_index(theoretical_lines,abs_waves[i])
    th = theoretical_lines[idx]
    if (np.abs(th-abs_waves[i])/th < 0.01):
        compared_lines.append(abs_waves[i])
for i in range(len(compared_lines)):
    print(f"{i}: {compared_lines[i]}")