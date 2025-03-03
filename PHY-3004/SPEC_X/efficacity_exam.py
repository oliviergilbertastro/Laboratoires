import numpy as np
import matplotlib.pyplot as plt
import json



energies = [13.945341724710236, 17.740602419774884, 59.48608752619144]
efficacity_Si_theoretical = [75.75301204819277, 48.94578313253012, 1.6566265060240823]
efficacity_CdTe_theoretical = [100,100,100]

relative_efficacity_theo = np.array(efficacity_Si_theoretical)/np.array(efficacity_CdTe_theoretical)

with open("PHY-3004/SPEC_X/infos.json") as json_file:
    infos = json.load(json_file)

Integrals_save = infos["Count_rates"]
efficacity_Si_experimental = [np.mean(Integrals_save["XR100CR_Si"][i]) for i in range(3)]
efficacity_Si_experimental_std = [np.std(Integrals_save["XR100CR_Si"][i]) for i in range(3)]
efficacity_CdTe_experimental = [np.mean(Integrals_save["XR100T_CdTe"][i]) for i in range(3)]
efficacity_CdTe_experimental_std = [np.std(Integrals_save["XR100T_CdTe"][i]) for i in range(3)]
print(efficacity_Si_experimental, efficacity_Si_experimental_std)
print(efficacity_CdTe_experimental, efficacity_CdTe_experimental_std)
relative_efficacity = np.array(efficacity_Si_experimental)/np.array(efficacity_CdTe_experimental)
relative_efficacity_std = [relative_efficacity[i]*np.sqrt((efficacity_Si_experimental_std[i]/efficacity_Si_experimental[i])**2+(efficacity_CdTe_experimental_std[i]/efficacity_CdTe_experimental[i])**2) for i in range(3)]
print(relative_efficacity, relative_efficacity_std)


print("*********************")
h_rel = efficacity_Si_experimental[0]/efficacity_Si_experimental[-1]
h_std = h_rel*np.sqrt((efficacity_Si_experimental_std[0]/efficacity_Si_experimental[0])**2+(efficacity_Si_experimental_std[-1]/efficacity_Si_experimental[-1])**2)
print(h_rel, h_std)
h_rel = efficacity_CdTe_experimental[0]/efficacity_CdTe_experimental[-1]
h_std = h_rel*np.sqrt((efficacity_CdTe_experimental_std[0]/efficacity_CdTe_experimental[0])**2+(efficacity_CdTe_experimental_std[-1]/efficacity_CdTe_experimental[-1])**2)
print(h_rel, h_std)


FWHMs = infos["FWHMs"]
FWHM_Si = [np.mean(FWHMs["XR100CR_Si"][i]) for i in range(3)]
FWHM_Si_std = [np.std(FWHMs["XR100CR_Si"][i]) for i in range(3)]
FWHM_CdTe = [np.mean(FWHMs["XR100T_CdTe"][i]) for i in range(3)]
FWHM_CdTe_std = [np.std(FWHMs["XR100T_CdTe"][i]) for i in range(3)]

print(FWHM_Si, FWHM_Si_std)
print(FWHM_CdTe, FWHM_CdTe_std)

# PLOT EFFICACITÉ
labels = ['13.95keV', '17.74keV', '59.54keV']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, relative_efficacity_theo, width, label='Théorique')
rects2 = ax.bar(x + width/2, relative_efficacity, width, label='Expérimental')
plt.errorbar(x + width/2, relative_efficacity, yerr=np.array(relative_efficacity_std)*1, fmt="o", capsize=10, color="black")
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Ratio efficacité Si/CdTe', fontsize=15)
#ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=15)
ax.yaxis.set_tick_params(labelsize=14)
ax.legend(fontsize=15)
fig.tight_layout()
plt.savefig("PHY-3004/SPEC_X/efficacite_comp.pdf")
plt.show()


# PLOT EFFICACITÉ
labels = ['13.95keV', '17.74keV', '59.54keV']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, FWHM_Si, width, label='Si')
rects2 = ax.bar(x + width/2, FWHM_CdTe, width, label='CdTe')
prop_cycle = plt.rcParams['axes.prop_cycle']
def_colors = prop_cycle.by_key()['color']
plt.errorbar(x - width/2, FWHM_Si, yerr=np.array(FWHM_Si_std)*1, fmt="o", capsize=10, color="black")
plt.errorbar(x + width/2, FWHM_CdTe, yerr=np.array(FWHM_CdTe_std)*1, fmt="o", capsize=10, color="black")
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('FWHM [# de canaux]', fontsize=15)
#ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=15)
ax.yaxis.set_tick_params(labelsize=14)
ax.legend(fontsize=15)
fig.tight_layout()
plt.savefig("PHY-3004/SPEC_X/resolution_comp.pdf")
plt.show()
