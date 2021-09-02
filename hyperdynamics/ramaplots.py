import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np

# fix colorbars
# add time suptitle

plt_params = {'font.size': 11,
    'axes.labelsize': 13,
    'axes.linewidth': 0.8,
    'axes.labelweight': 'normal',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 1.0,
    'legend.fontsize': 8,
    'legend.borderpad': 0.3,
    'legend.handletextpad': 0.5,
    'legend.labelspacing': 0.2,
    'legend.handlelength': 1.0,
    'legend.markerscale': 0.8,
    'savefig.facecolor': 'white',
    'errorbar.capsize': 1}

plt.rcParams.update(plt_params)

def scat(fig, ax, A,col=None,m=0):
    S = 8
    if col is not None and m == 0:
        m = max(col)

    j = ax.scatter(A[:,0],A[:,1], s=S, c=col, vmin=-1, vmax=m)
    if col is not None:
        if type(m) == int and m < 1000:
            ticks = [i for i in range(-1,m+1)]
        else:
            ticks = None
        #fig.subplots_adjust(right=0.93)
        #cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
        #fig.colorbar(j, cax=cbar_ax, ticks=ticks)
    return fig, ax

def setup_ax(ax):
    ax.set_aspect(1)
    ax.set_xlim([-180,180])
    ax.set_ylim([-180,180])
    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.yaxis.set_major_locator(MultipleLocator(60))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.set_xlabel(r'$\phi$ (degrees)')
    ax.set_ylabel(r'$\psi$ (degrees)')

def setup_one(time, figsize=(6,6)):
    fig, ax = plt.subplots(1,figsize=figsize)
    setup_ax(ax)
    ax.set_title("Trajectory at time {} ps".format(time))

    return fig, ax

def setup_two(figsize=(10,5)):
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=figsize)

    for ax in [ax1,ax2]:
        setup_ax(ax)

    ax2.set_ylabel('')
    ax1.set_title("DBSCAN clustering")
    ax2.set_title("SVM boundary and values")

    fig.subplots_adjust(wspace=0.4, right=0.90)
    #bbox_ax = ax1.get_position()
    #cax1 = fig.add_axes([bbox_ax.x1+0.01, bbox_ax.y0, 0.02, bbox_ax.y1-bbox_ax.y0])
    #bbox_ax = ax2.get_position()
    #cax2 = fig.add_axes([bbox_ax.x1+0.01, bbox_ax.y0, 0.02, bbox_ax.y1-bbox_ax.y0])

    return fig, [ax1, ax2]
    
def scat2(fig, axes, A,col1=None,col2=None,t=None,m=0):
    S = 8
    if m == 0 and col1 is not None:
        if col2 is not None:
            m = max(max(col1),max(col2))
        else:
            m = max(col1)

    ax1, ax2 = axes
    ax1.scatter(A[:,0],A[:,1], s=S, c=col1, vmin=-1, vmax=m)
    j = ax2.scatter(A[:,0],A[:,1], s=S, c=col2, vmin=-1, vmax=m)
    if col1 is not None:
        ticks = [i for i in range(-1,m+1)]
        fig.subplots_adjust(right=0.93)
        cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
        fig.colorbar(j, cax=cbar_ax, ticks=ticks)
    if t != None:
        plt.suptitle('time ' + str(t))

    return fig, ax1, ax2

def scat2dists(fig, axes, A, labels, X=None, Y=None, Z=None, t=None):
    S = 6
    ax1, ax2 = axes

    # Plot trajectory with cluster labels
    if labels is not None:
        sc1 = ax1.scatter(A[:,0],A[:,1], s=S, c=labels, vmin=-1)

        # Add colorbar
        bbox_ax = ax1.get_position()
        max_region = max(labels) 
        bounds = np.arange(-1.5, max_region+1.5,1)
        cax1 = fig.add_axes([bbox_ax.x1+0.01, bbox_ax.y0, 0.02, bbox_ax.y1-bbox_ax.y0])
        cbar1 = fig.colorbar(sc1, cax=cax1, boundaries=bounds, ticks=range(-1,max_region+1))

    # Plot SVM contour
    if X is not None:
        sc2 = ax2.scatter(A[:,0],A[:,1], s=S, c=labels, vmin=-1)
        j = ax2.contourf(X,Y,Z,np.arange(0,10.5,0.5))
        #m = max(Z.flatten())
        #levels = m/10
        #j = ax2.contourf(X,Y,Z,np.arange(0,m+levels,levels))

        # Add colorbar
        bbox_ax = ax2.get_position()
        cax2 = fig.add_axes([bbox_ax.x1+0.01, bbox_ax.y0, 0.02, bbox_ax.y1-bbox_ax.y0])
        cbar2 = fig.colorbar(j, cax=cax2, label='"Distance"', ticks=np.arange(0,11,1))

    if t != None:
        fig.text(0.45, 0.91, "Time: {:6.1f} ps".format(t), fontsize=13)
    return 

def scat3angs(A, col=None, t=None, m=0):
    S = 8
    if col is not None and m == 0:
        m = max(col)
    fig, axs = plt.subplots(1,3, figsize=(15,5))
    for ax in axs:
        ax.set_xlim([-180,180])
        ax.set_ylim([-180,180])
    axs[0].scatter(A[:,0],A[:,1], s=S, c=col, vmin=-1, vmax=m)
    axs[1].scatter(A[:,2],A[:,3], s=S, c=col, vmin=-1, vmax=m)
    j = axs[2].scatter(A[:,4],A[:,5], s=S, c=col, vmin=-1, vmax=m)
    if type(m) == int:
        ticks = [i for i in range(-1,m+1)]
    else:
        ticks = None
    if col is not None:
        fig.subplots_adjust(right=0.93)
        cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
        fig.colorbar(j, cax=cbar_ax, ticks=ticks)
    if t != None:
        plt.suptitle('time ' + str(t))
    return


def scat2angs(A,col1=None):
    S = 8
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))
    ax1.scatter(A[:,0],A[:,1], s=S, c=col1)
    ax2.scatter(A[:,2],A[:,3], s=S, c=col1)
    return



