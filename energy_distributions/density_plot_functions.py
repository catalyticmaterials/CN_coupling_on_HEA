import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

#get color map
cmap = truncate_colormap(plt.get_cmap("hot_r"),maxval=0.75)


def probability_density_plot(x,y,return_hists=False,figure=None,colorbar=True,N=10000,vmax=None):
        if figure is not None:
              fig,ax = figure
        else:
            fig,ax = plt.subplots(dpi=400,figsize=(8,6))
        
        # Bin data
        hist_x,xedges = np.histogram(x,bins=100)
        hist_y,yedges = np.histogram(y,bins=100)
        
        # Convert to probabilities
        prob_x = hist_x/N /np.diff(xedges)
        prob_y = hist_y/N /np.diff(yedges)

        # Make into a 2D probability distribution
        prob_density = np.outer(prob_x,prob_y)

        # Pad probability density function with empty bins to get smoother plotting
        prob_density_pad = np.pad(prob_density,pad_width=((2,2),(2,2)),mode='constant',constant_values=0)

        # Get bin widths
        x_bin_length = abs(xedges[1]-xedges[0])
        y_bin_length = abs(yedges[1]-yedges[0])

        # Plot distribution
        im=ax.imshow(prob_density_pad.T,norm=LogNorm(vmin=1e-4,vmax=vmax),cmap=cmap,interpolation='gaussian',origin='lower',extent=[xedges[0]-2*x_bin_length, xedges[-1]+2*x_bin_length,yedges[0]-2*y_bin_length,yedges[-1]+2*y_bin_length],aspect="auto")
        
        # Prepare ax for colorbar
        the_divider = make_axes_locatable(ax)
        color_axis = the_divider.append_axes("right", size="5%", pad='2%')

        # Add colorbar
        if colorbar: 
            cbar=plt.colorbar(im,ticks=[1e-4,0.001,0.01,0.1,1,10],cax=color_axis)
            cbar.ax.set_yticklabels(["0","$10^{-3}$","$10^{-2}$","$10^{-1}$","1","10"])
            cbar.ax.set_ylabel("Probability Density")
        if return_hists:
            return fig,ax,(hist_x,xedges),(hist_y,yedges), prob_density
        else:
            return fig,ax
