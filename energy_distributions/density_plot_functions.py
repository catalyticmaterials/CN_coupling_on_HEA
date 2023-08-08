import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

#get color map
cmap = truncate_colormap(plt.get_cmap("hot_r"),maxval=0.75)


def probability_density_plot(x,y,return_hists=False):
        fig,ax = plt.subplots(dpi=400,figsize=(8,6))
        #hist, xedges, yedges = np.histogram2d(x, y,bins=25,density=True)
        #areas = np.matmul(np.array([np.diff(xedges)]).T, np.array([np.diff(yedges)]))
        #hist=hist.T
        #prob_density = hist*areas
        hist_x,xedges = np.histogram(x,bins=200,density=True)
        hist_y,yedges = np.histogram(y,bins=200,density=True)
        
        areas = np.matmul(np.array([np.diff(xedges)]).T, np.array([np.diff(yedges)]))
        
        prob_density = np.outer(hist_x,hist_y)*areas
        # print(prob_density)
        im=ax.imshow(prob_density.T,norm=LogNorm(vmin=1e-8,vmax=1.0),cmap=cmap,interpolation='gaussian',origin='lower',extent=[xedges[0], xedges[-1],yedges[0],yedges[-1]],aspect="auto")
        cbar=plt.colorbar(im,ticks=[1e-8,1e-5,1e-4,0.001,0.01,0.1,1])
        cbar.ax.set_yticklabels(["0","$10^{-5}$","$10^{-4}$","$10^{-3}$","$10^{-2}$","$10^{-1}$","1"]) 
        cbar.ax.set_ylabel("Probability Density")
        if return_hists:
            return fig,ax,(hist_x,xedges),(hist_y,yedges), prob_density
        else:
            return fig,ax
