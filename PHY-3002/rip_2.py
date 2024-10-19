import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl

pos_maxima_241_Hz = [-34.39,-28.77,-22.73,-16.98,-10.59,-4.21,0,8.85,15.23,21.55,28.43,35.66,43.94]
pos_maxima_341_Hz = [-41.73,-35.82,-29.73,-22.67,-15.17,-7.31,0,8.75,17.40,26.13,36.02,46.87,58.70]
pos_maxima_441_Hz = [ -40.81,-34.05,-26.62,-18.36,-9.60,0,9.93,19.70,31.22,43.90,58.34]


pos_maximas = [pos_maxima_241_Hz,pos_maxima_341_Hz,pos_maxima_441_Hz]

m0_to_plan = 176 #mm
L = 1855 #mm
theta = np.arctan(m0_to_plan/L) #rad
lam = 632.8E-6 #mm

def alpha_n(pos_max):
    angles = np.arctan(pos_max/L)
    return angles

Lambdas = []

for k, pos_maxima in enumerate(pos_maximas):
    if pos_maxima == pos_maxima_441_Hz:
        maxima = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
    else:
        maxima = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]

    # On flip les positions parce qu'on est cons:
    flipped_pos_maxima = []
    for i in range(len(pos_maxima)):
        flipped_pos_maxima.append(-pos_maxima[len(pos_maxima)-1-i])
    pos_maxima = flipped_pos_maxima
    pos_maxima = np.array(pos_maxima)+m0_to_plan #mm


    def line(n, Lambda):
        return np.cos(theta)-n*lam/Lambda


    alpha_ns = alpha_n(pos_maxima)
    
    res = curve_fit(line, maxima, np.cos(alpha_ns))[0]


    print("Lambda =", res[0], "mm")
    Lambdas.append(res[0])

    x_fit = np.linspace(-6,6,100)
    y_fit = line(x_fit, Lambda=res[0])
    #mpl.rc('text', usetex=True)
    plt.figure()
    ax1 = plt.gca()
    plt.plot(maxima, np.cos(alpha_ns), "o")
    plt.plot(x_fit, y_fit, "--")
    plt.xlabel(r"Ordre $n$", fontsize=17)
    plt.ylabel(r"$\cos(\alpha_n)$", fontsize=17)
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)


    col_labels=[r'$n$',r'$h_n$ $\mathrm{[mm]}$', r'$\alpha_n$ $\mathrm{[^\circ]}$']
    table_vals=np.array([np.array(maxima, dtype=int),np.around(pos_maxima, decimals=1),np.around(alpha_ns, decimals=3)]).T
    # the rectangle is where I want to place the table
    the_table = plt.table(cellText=table_vals,
                    colWidths = [0.05,0.1,0.07],
                    cellLoc = "center",
                    #rowLabels=row_labels,
                    colLabels=col_labels,
                    loc='upper right')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1.3, 1)



    plt.subplots_adjust(left=0.188, bottom=0.129, right=0.967, top=0.955, wspace=0.2, hspace=0.2)
    plt.savefig(f"PHY-3002/Graph_RIP/graph_{k}.png")
    plt.show()


frequences = np.array([241.8,341.8,441.8])
rho_eau = 997 # kg/m^3
def Lambda_relation_capillarite(f, tension_sup):
    return (tension_sup/rho_eau*2*np.pi/(f**2))**(1/3)*1000

res = curve_fit(Lambda_relation_capillarite, frequences, Lambdas)
print("Tension superficielle =", res, "N/m")
res = res[0]
x_fit = np.linspace(200,500,1000)
y_fit = Lambda_relation_capillarite(x_fit, tension_sup=res[0])
y_theorique = Lambda_relation_capillarite(x_fit, tension_sup=0.07199)


# Create the plot axes:
fig = plt.figure(figsize=(7.5,7.5))
gs = mpl.gridspec.GridSpec(10, 10, wspace=0.0, hspace=0.0)    
data_ax = fig.add_subplot(gs[0:7, 0:10]) # Sérsic index histogram
res_ax = fig.add_subplot(gs[7:10, 0:10], sharex=data_ax) # B/T histogram
data_ax.plot(x_fit, y_fit, "-", linewidth=2, label="Fit pratique", color="red")
data_ax.plot(x_fit, y_theorique, "--", linewidth=3, label="Théorique", color="blue")
data_ax.plot(frequences, Lambdas, "o", label="Données", color="black")
data_ax.set_xlabel(r"Fréquence $\mathrm{[Hz]}$", fontsize=17)
data_ax.set_ylabel(r"$\Lambda$ $\mathrm{[mm]}$", fontsize=17)
data_ax.xaxis.set_tick_params(labelsize=14)
data_ax.yaxis.set_tick_params(labelsize=14)
data_ax.legend(fontsize=13)

res_ax.plot(frequences, np.abs(Lambdas-Lambda_relation_capillarite(frequences, tension_sup=0.07199))/Lambda_relation_capillarite(frequences, tension_sup=0.07199)*100, "o", linewidth=3, label="Théorique", color="black")
#res_ax.plot(x_fit, np.abs(y_fit-y_theorique)/y_theorique*100, "-", linewidth=3, label="Théorique", color="grey")
res_ax.set_xlabel(r"Fréquence $\mathrm{[Hz]}$", fontsize=17)
res_ax.set_ylabel(r"Écart $\mathrm{[\%]}$", fontsize=17)
res_ax.xaxis.set_tick_params(labelsize=14)
res_ax.yaxis.set_tick_params(labelsize=14)

plt.subplots_adjust(left=0.133, bottom=0.129, right=0.967, top=0.955, wspace=0.2, hspace=0)
plt.savefig(f"PHY-3002/Graph_RIP/graph_f_lambda.png")
plt.show()


if False:
    x_fit = np.linspace(-6,6,100)
    y_fit = line(x_fit, Lambda=res[0])
    plt.plot(maxima, alpha_ns, "o")
    plt.plot(x_fit, np.arccos(y_fit), "--")
    plt.xlabel(r"Ordre $m$", fontsize=17)
    plt.ylabel(r"$\alpha_n$", fontsize=17)
    plt.show()
    pass