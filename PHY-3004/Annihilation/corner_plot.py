import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import numpy as np
import seaborn as sns


def print_color(message, color="yellow", **kwargs):
    """print(), but with a color option"""
    possible_colors = ["black","red","green","yellow","blue","magenta","cyan","white"]
    if color == None or color == "grey":
        color = "0"
    elif type(color) == str:
        color = color.lower()
        if color in possible_colors:
            color = str(possible_colors.index(color)+30)
        else:
            print(f"Color '{color}' not implemented, defaulting to grey.\nPossible colors are: {['grey']+possible_colors}")
            color = "0"
    else:
        raise ValueError(f"Parameter 'header_color' needs to be a string.")
    print(f"\x1b[{color}m{message}\x1b[0m", **kwargs)



def myCornerPlot(data, labels=None, bw_adjust=1, units=None, fontsize=15, smoothness=6, linewidth=3, extremums=None, background_colors=["#f0f0f0","#969696","#252525"], levels=4, markersize=8, columns_compare=None, save_plot=None):
    """
    data should be [data_set1, data_set2, ...] each containing multiple parameters
    """
    for i in range(len(data)-1):
        assert len(data[i]) == len(data[i+1])

    # labels are now required (just put empty ones if you don't want them to appear)
    if extremums is None:
        extremums = {}
    for label in labels:
        if label not in extremums:
            extremums[label] = None
    
    
    # Create the plot axes:
    fig = plt.figure(figsize=(10,8))
    plot_size = len(data[0])
    if labels is not None:
        assert plot_size == len(labels)
    if units is not None:
        assert plot_size == len(units)
    hist_axes = []
    corner_axes = []
    for i in range(plot_size):
        hist_axes.append(plt.subplot(plot_size,plot_size,i*plot_size+i+1))
        corner_axes.append([])
        for k in range(plot_size-(i+1)):
            if i == 0:
                corner_axes[i].append(plt.subplot(plot_size,plot_size,(i+k+1)*plot_size+(i+1),sharex=hist_axes[i]))
            else:
                corner_axes[i].append(plt.subplot(plot_size,plot_size,(i+k+1)*plot_size+(i+1),sharex=hist_axes[i],sharey=corner_axes[i-1][k+1]))
            if k != plot_size-(i+1)-1:
                corner_axes[i][k].get_xaxis().set_visible(False)
            if i != 0:
                corner_axes[i][k].get_yaxis().set_visible(False)
            corner_axes[i][k].xaxis.set_tick_params(labelsize=fontsize-2)
            corner_axes[i][k].yaxis.set_tick_params(labelsize=fontsize-2)
        if i == plot_size-1:
            hist_axes[i].get_yaxis().set_visible(False)
            hist_axes[i].xaxis.set_tick_params(labelsize=fontsize-2)
        else:
            hist_axes[i].get_xaxis().set_visible(False)
            hist_axes[i].get_yaxis().set_visible(False)

    # Show data in each plot:

    
    #Plot kernel histograms:
    for i in range(plot_size):
        if labels is not None:
            hist_axes[i].set_title(labels[i], fontsize=fontsize)
        x_min, x_max = np.min(data[0][i]), np.max(data[0][i])
        for j in range(len(data)):
            x_min = np.min(data[j][i]) if x_min > np.min(data[j][i]) else x_min
            x_max = np.max(data[j][i]) if x_max < np.max(data[j][i]) else x_max
        if extremums[labels[i]] is not None:
            x_min, x_max = extremums[labels[i]]
        for j in range(len(data)):
            print(j)
            X_plot = np.linspace(x_min, x_max, 1000)[:,np.newaxis]
            bandwidth = np.abs(x_max-x_min)/smoothness
            kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(data[j][i][:,np.newaxis])
            log_dens = kde.score_samples(X_plot)
            hist_axes[i].fill_between(X_plot[:, 0], np.exp(log_dens), fc=["blue","red","orange","green"][j%4], alpha=0.4)


 
    for i in range(plot_size):
        for k in range(len(corner_axes[i])):
            for j in range(len(data)):
                corner_axes[i][k].plot(data[j][i], data[j][i+k+1], ["o","*","*"][j%3], color=["blue","red","red"][j%3], markersize=[markersize,markersize-1,markersize-1][j%3])

    # Make units labels and set axis limits:
    if units is None:
        units = [" " for i in range(plot_size)]
    for i in range(plot_size):
        if i < plot_size-1:
            if i > 0:
                corner_axes[0][i-1].set_ylabel(labels[i]+"\n"+units[i], fontsize=fontsize-1)
                if extremums[labels[i]] is not None:
                    corner_axes[0][i-1].set_ylim(*extremums[labels[i]])
            corner_axes[i][-1].set_xlabel(labels[i]+"\n"+units[i], fontsize=fontsize-1)
            if extremums[labels[i]] is not None:
                corner_axes[i][-1].set_xlim(*extremums[labels[i]])
        else:
            corner_axes[0][i-1].set_ylabel(labels[i]+"\n"+units[i], fontsize=fontsize-1)
            if extremums[labels[i]] is not None:
                corner_axes[0][i-1].set_ylim(*extremums[labels[i]])
            hist_axes[i].set_xlabel(labels[i]+"\n"+units[i], fontsize=fontsize-1)
            if extremums[labels[i]] is not None:
                hist_axes[i].set_xlim(*extremums[labels[i]])
    
    plt.subplots_adjust(left=0.095, bottom=0.1, right=0.99, top=0.955, wspace=0, hspace=0)
    if save_plot is not None:
        plt.savefig(f"{save_plot}.png")
    plt.show()
    return

RESOLUTIONS = [100,75,50,25]
datas = []
for resolution in RESOLUTIONS:
    data = np.loadtxt(f"PHY-3004/Annihilation/params_{resolution}.txt")
    datas.append(data)
datas = np.array(datas)
print(data.shape)
myCornerPlot([datas[i].T for i in range(len(datas))], labels=[r"$\sigma$", r"Scale", r"$\mu$", r"$+C$"])