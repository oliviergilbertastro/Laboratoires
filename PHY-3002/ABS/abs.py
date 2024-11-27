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

profile = profile[1040:2749]
lines_px -= 1040

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

# Find the absorption lines
from scipy.signal import boxcar, find_peaks
def get_absorption_lines_indices(valeurs: np.array,
                                      hauteur_minimum: int = None,
                                      distance_minimum: int = None):
    peaks, _ = find_peaks(-valeurs, height=hauteur_minimum, distance=distance_minimum)
    return peaks

abs_lines_px = get_absorption_lines_indices(profile, distance_minimum=24)

for i in range(len(abs_lines_px)):
    print(f"{i}: {convert_px_to_lambda(abs_lines_px[i], res[0], res[1])}")

ax1 = plt.subplot(111)
plt.plot(profile, color="black")
plt.plot(abs_lines_px, profile[abs_lines_px], "o", color="blue")
plt.plot(lines_px, profile[lines_px], "o", color="red")
ax1.set_xlabel("pixel", fontsize=15)
ax1.set_ylabel("intensité", fontsize=15)
plt.show()

ax1 = plt.subplot(111)
longueurdondes = convert_px_to_lambda(range(len(profile)),res[0],res[1])
plt.plot(longueurdondes, profile, color="black")
plt.plot(convert_px_to_lambda(abs_lines_px,res[0],res[1]), profile[abs_lines_px], "o", color="blue")
plt.plot(convert_px_to_lambda(lines_px,res[0],res[1]), profile[lines_px], "o", color="red")
ax1.set_xlabel("$\lambda$ [nm]", fontsize=15)
ax1.set_ylabel("intensité", fontsize=15)
plt.show()

theoretical_lines = np.array([508.674,509.529,510.433,511.388,512.396,513.458,514.576,515.751,516.984,518.278,519.634,521.052,522.535,524.083,525.699,527.383,529.137,530.962,532.860,534.831,536.877,538.999,541.199,543.478,545.838,548.279,547.520,550.803,549.853,553.411,552.268,556.106,554.767,558.888,557.351,561.759,560.022,564.721,562.781,567.775,565.631,570.924,568.572,574.169,571.606,574.736,577.512,577.952])
experimental_lines = np.array([508.762,509.411,510.522,511.403,512.361,513.300,514.542,515.696,516.936,518.219,519.544,520.955,522.366,523.905,525.488,527.112,528.823,530.576,532.457,534.381,536.433,539.042,540.880,543.959,545.199,548.533,547.715,551.129,550.110,553.776,552.618,556.493,555.222,559.280,557.712,562.180,560.456,565.002,563.129,567.919,565.968,571.037,568.764,574.128,571.638,574.738,577.568,577.938])
def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

experimental_lines_indices = []
for i in range(len(experimental_lines)):
    experimental_lines_indices.append(find_nearest_index(longueurdondes,experimental_lines[i]))
experimental_lines_indices = np.array(experimental_lines_indices)




nus_bleus = []
nus_prime_bleus = []
exp_lines_bleus = experimental_lines[:25]
for i in range(len(exp_lines_bleus)):
    nus_bleus.append(0)
    nus_prime_bleus.append(49-i)
nus_verts_0 = []
nus_verts_1 = []
nus_prime_verts_0 = []
nus_prime_verts_1 = []
exp_lines_verts = experimental_lines[25:]
exp_lines_verts_0 = []
exp_lines_verts_1 = []
for i in range(len(exp_lines_verts)-3):
    if i % 2 == 0:
        nus_verts_0.append(0)
        nus_prime_verts_0.append(int(24-i/2))
        exp_lines_verts_0.append(exp_lines_verts[i])
    else:
        nus_verts_1.append(1)
        nus_prime_verts_1.append(int(27-(i-1)/2))
        exp_lines_verts_1.append(exp_lines_verts[i])

        
nus_verts_1.append(1)
nus_verts_0.append(0)
nus_verts_1.append(1)
nus_prime_verts_1.append(17)
nus_prime_verts_0.append(14)
nus_prime_verts_1.append(16)
exp_lines_verts_1.append(exp_lines_verts[-3])
exp_lines_verts_0.append(exp_lines_verts[-2])
exp_lines_verts_1.append(exp_lines_verts[-1])
print(nus_bleus)
print(nus_prime_bleus)
print(nus_verts_0)
print(nus_prime_verts_0)
print(nus_verts_1)
print(nus_prime_verts_1)



fig = plt.figure(figsize=(14,10))
ax1 = plt.subplot(111)
profile = profile/np.max(profile)
plt.plot(longueurdondes, profile, color="black")
plt.plot(longueurdondes[experimental_lines_indices], profile[experimental_lines_indices], "o", color="blue")
plt.plot(convert_px_to_lambda(lines_px,res[0],res[1]), profile[lines_px], "o", color="red")
ax1.set_xlabel("$\lambda$ [nm]", fontsize=15)
ax1.set_ylabel("Intensité $I/I_0$", fontsize=15)

col_labels1=[r'$\nu$',r"$\nu$'","$\lambda$ [nm]"]
table_vals=np.array([nus_bleus,nus_prime_bleus,exp_lines_bleus]).T
# the rectangle is where I want to place the table
the_table1 = plt.table(cellText=table_vals,
                colWidths = [0.05,0.05,0.08],
                cellLoc = "center",
                #rowLabels=row_labels,
                colLabels=col_labels1,
                #loc='upper center',
                bbox=(0.35,0.28,0.2,0.7))
the_table1.auto_set_font_size(False)
the_table1.set_fontsize(10)
the_table1.scale(1, 1)

height_per_box = 0.7/26

col_labels1=[r'$\nu$',r"$\nu$'","$\lambda$ [nm]"]
table_vals=np.array([nus_verts_0,nus_prime_verts_0,exp_lines_verts_0]).T
# the rectangle is where I want to place the table
the_table1 = plt.table(cellText=table_vals,
                colWidths = [0.05,0.05,0.08],
                cellLoc = "center",
                #rowLabels=row_labels,
                colLabels=col_labels1,
                #loc='upper center',
                bbox=(0.575,0.28+14*height_per_box,0.2,height_per_box*(len(nus_verts_0)+1)))
the_table1.auto_set_font_size(False)
the_table1.set_fontsize(10)
the_table1.scale(1, 1)

col_labels1=[r'$\nu$',r"$\nu$'","$\lambda$ [nm]"]
table_vals=np.array([nus_verts_1,nus_prime_verts_1,exp_lines_verts_1]).T
# the rectangle is where I want to place the table
the_table1 = plt.table(cellText=table_vals,
                colWidths = [0.05,0.05,0.08],
                cellLoc = "center",
                #rowLabels=row_labels,
                colLabels=col_labels1,
                #loc='upper center',
                bbox=(0.775,0.28+13*height_per_box,0.2,height_per_box*(len(nus_verts_1)+1)))
the_table1.auto_set_font_size(False)
the_table1.set_fontsize(10)
the_table1.scale(1, 1)
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)
plt.savefig(f"PHY-3002/ABS/lignes_absorption.pdf")
plt.show()





# Correlation plot
fig = plt.figure(figsize=(7,7))
ax1 = plt.subplot(111)
plt.plot([lines_nm[0], lines_nm[-1]], [lines_nm[0], lines_nm[-1]], "--", linewidth=2, color="red")
plt.plot(theoretical_lines, experimental_lines, ".", color="black")
for i in range(4):
    plt.plot(lines_nm[i], lines_nm[i], "o", color=["blue","green","lime","lime"][i])
ax1.set_xlabel("$\lambda$ théorique [nm]", fontsize=15)
ax1.set_ylabel("$\lambda$ expérimental [nm]", fontsize=15)
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)
plt.savefig(f"PHY-3002/ABS/correlation_plot.pdf")
plt.show()


# Error percent
err_per = []
for i in range(len(theoretical_lines)):
    err_per.append(np.abs(theoretical_lines[i]-experimental_lines[i])/theoretical_lines[i]*100)

fig = plt.figure(figsize=(10,7))
ax1 = plt.subplot(111)
plt.plot(theoretical_lines, err_per, ".", color="black")
ax1.set_xlabel("$\lambda$ théorique [nm]", fontsize=15)
ax1.set_ylabel("Erreur [%]", fontsize=15)
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)
#plt.savefig(f"PHY-3002/ABS/error_plot.pdf")
plt.show()