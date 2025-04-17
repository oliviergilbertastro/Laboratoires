import numpy as np

background = (6+16+10) # counts
bg_time = (300+600+900) # s
N_c = (303+209) # counts
N_c_time = (5400+3600) # s

unc_bg = np.sqrt(background)
unc_N_c = np.sqrt(N_c)

background /= bg_time
unc_bg /= bg_time

N_c /= N_c_time
unc_N_c /= N_c_time

coincidences = N_c - background
unc_coincidences = np.sqrt(unc_N_c**2+unc_bg**2)

print(f"{coincidences} \pm {unc_coincidences} counts/s")