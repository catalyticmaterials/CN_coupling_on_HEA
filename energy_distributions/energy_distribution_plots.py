import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from ase.db import connect
import sys
sys.path.append("..")
from scripts.surface import initiate_surface, predict_energies
from scripts import metal_colors


# Define metals in the considered alloys
metals = ['Ag','Au', 'Cu', 'Pd','Pt']

single_metal_colors = [metal_colors[metal] for metal in metals]

# Get adsorption energies on single element
db_path = "../databases"
with connect(f'{db_path}/single_element_slabs_out.db') as db_slab,\
    connect(f'{db_path}/single_element_H_out.db') as db_H,\
        connect(f'{db_path}/single_element_CO_out.db') as db_CO,\
            connect(f'{db_path}/single_element_NO_fcc_out.db') as db_NO:
        
        
        E_ref_H = db_H.get(metal='Cu').energy - db_slab.get(metal='Cu').energy -0.1 + 0.1
        E_ref_CO = db_CO.get(metal='Cu').energy - db_slab.get(metal='Cu').energy +0.18 + 0.4
        E_ref_NO = db_NO.get(metal='Cu').energy - db_slab.get(metal='Cu').energy +0.45 + 0.71 
                
        H_metal_energies=[db_H.get(metal=m).energy - db_slab.get(metal=m).energy - E_ref_H for m in metals]
        CO_metal_energies=[db_CO.get(metal=m).energy - db_slab.get(metal=m).energy - E_ref_CO for m in metals]
        NO_metal_energies=[db_NO.get(metal=m).energy - db_slab.get(metal=m).energy - E_ref_NO for m in metals]

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



# Loop through selected compositions
for composition in [np.array([0.2,0.2,0.2,0.2,0.2]),np.array([0.1,0.1,0.6,0.1,0.1]),np.array([0.0,0.5,0.5,0.0,0.0])]:

    alloy_list = [metals[i] + str(int(composition[i]*100)) for i in range(len(metals)) if composition[i]>0]
    alloy = "".join(alloy_list)
    alloy_sub_list = [metals[i] + '_{' + str(int(composition[i]*100)) + '}' for i in range(len(metals)) if composition[i]>0]
    alloy_sub = '$' + ''.join(alloy_sub_list) + '$'
    
    #Initiate surface
    #np.random.seed(1)
    #surface = initiate_surface(composition,metals)

    #Predict energies
    #NO_energies,NO_site_ids=predict_energies(surface, "NO_fcc", metals) 
    #CO_energies,CO_site_ids=predict_energies(surface, "CO", metals)
    #H_energies,H_site_ids=predict_energies(surface, "H", metals)

    # Load energies
    NO_energies = np.loadtxt(f'{alloy}/NO_fcc_energies.csv',delimiter=',',usecols=(0))
    CO_energies = np.loadtxt(f'{alloy}/CO_energies.csv',delimiter=',',usecols=(0))
    H_energies = np.loadtxt(f'{alloy}/H_energies.csv',delimiter=',',usecols=(0))
    H_energies = H_energies[H_energies<10]

    #Correct to adsorption energies
    NO_energies-=0.71
    CO_energies-=0.4
    H_energies-=0.1

    
    #CO vs H plot    
    fig,ax=probability_density_plot(H_energies, CO_energies)
    ax.set_xlabel("\u0394$E_{pred}^{*H}$ [eV]")
    ax.set_ylabel("\u0394$E_{pred}^{*CO}$ [eV]")
    plt.title(alloy_sub)
    for i, metal in enumerate(metals):
        ax.scatter(H_metal_energies[i],CO_metal_energies[i],color=metal_colors[metal],edgecolors='k',label=metal)
    ax.legend(loc=2)
    #for (x,y,s) in zip(H_metal_energies,CO_metal_energies,metals):
        #ax.text(x,y,s)
    plt.tight_layout()
    plt.savefig(f'{alloy}/Energy_dist_COvsH.png')
    

    #NO vs H plot    
    fig,ax=probability_density_plot(H_energies, NO_energies)
    ax.set_xlabel("\u0394$E_{pred}^{*H}$ [eV]")
    ax.set_ylabel("\u0394$E_{pred}^{*NO}$ [eV]")
    plt.title(alloy_sub)
    for i, metal in enumerate(metals):
        ax.scatter(H_metal_energies[i],NO_metal_energies[i],color=metal_colors[metal],edgecolors='k',label=metal)
    # for (x,y,s) in zip(H_metal_energies,NO_metal_energies,metals):
    #     ax.text(x,y,s)
    ax.legend(loc=2)
    plt.tight_layout()
    plt.savefig(f'{alloy}/Energy_dist_NOvsH.png')
        
    #NO vs CO
    fig,ax,(hist_x,xedges),(hist_y,yedges), prob_density=probability_density_plot(CO_energies, NO_energies,return_hists=True)
    ax.set_ylabel("\u0394$E_{pred}^{*NO}$ [eV]")
    ax.set_xlabel("\u0394$E_{pred}^{*CO}$ [eV]")
    plt.title(alloy_sub)
    for i, metal in enumerate(metals):
        ax.scatter(CO_metal_energies[i],NO_metal_energies[i],color=metal_colors[metal],edgecolors='k',label=metal)
    # for (x,y,s) in zip(CO_metal_energies,NO_metal_energies,metals):
    #     ax.text(x+0.01,y+0.01,s)
    ax.legend(loc=2)


    #integrate
    mask_x = lambda x: (x>=(-1.1))*(x<=(-0.4))
    mask_y = lambda y: (y>=(-1.3))*(y<=(-0.71))

    x_centers = (xedges[1:] + xedges[:-1])/2
    y_centers = (yedges[1:] + yedges[:-1])/2

    x_centers_id=np.where(mask_x(x_centers))[0]
    y_centers_id=np.where(mask_y(y_centers))[0]

    prob_density_urea = np.sum(prob_density.T[y_centers_id[0]:y_centers_id[-1],x_centers_id[0]:x_centers_id[-1]])

    ax.fill_between([-1.1,-0.4],[-1.3,-1.3],y2=[-0.71,-0.71],color="tab:blue",alpha=0.2)

    ylim=ax.get_ylim()
    xlim=ax.get_xlim()

    plt.plot([-0.4,-0.4],ylim,c="k",ls="--",alpha=0.6)
    plt.plot(xlim,[-0.71,-0.71],c="k",ls="--",alpha=0.6)

    plt.plot([-1.1,-1.1],ylim,c="b",ls="--")
    plt.plot(xlim,[-1.3,-1.3],c="b",ls="--")

    ax.set_ylim(ylim)
    prob=(sum(mask_x(CO_energies))/10000) * (sum(mask_y(NO_energies))/10000)
    ax.text((-0.4-1.1)/2,(-1.3-0.71)/2,f"Urea \nProbability: {prob:1.3f}",c="b",alpha=0.8,ha="center",va="center")
    
    plt.tight_layout()
    plt.savefig(f'{alloy}/Energy_dist_NOvsCO.png')
    