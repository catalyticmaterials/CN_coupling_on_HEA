import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from ase.db import connect

# Define metals in the considered alloys
metals = ['Ag','Au', 'Cu', 'Pd','Pt']

db_path = "../../databases"
with connect(f'{db_path}/single_element_slabs_out.db') as db_slab,\
	 connect(f'{db_path}/single_element_H_out.db') as db_H,\
         connect(f'{db_path}/single_element_CO_out.db') as db_CO,\
             connect(f'{db_path}/single_element_NO_angled_out.db') as db_NO:
         
        
         E_ref_H = db_H.get(metal='Cu').energy - db_slab.get(metal='Cu').energy -0.1 + 0.1
         E_ref_CO = db_CO.get(metal='Cu').energy - db_slab.get(metal='Cu').energy +0.18 + 0.4
         E_ref_NO = db_NO.get(metal='Cu').energy - db_slab.get(metal='Cu').energy +0.45 + 0.71 
                 
         H_energies=[db_H.get(metal=m).energy - db_slab.get(metal=m).energy - E_ref_H for m in metals]
         CO_energies=[db_CO.get(metal=m).energy - db_slab.get(metal=m).energy - E_ref_CO for m in metals]
         NO_energies=[db_NO.get(metal=m).energy - db_slab.get(metal=m).energy - E_ref_NO for m in metals]
         
        

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


#get color map
cmap = truncate_colormap(plt.get_cmap("hot_r"),maxval=0.75)


CO=np.loadtxt("CO_energies.csv",usecols=(0),delimiter=" ") - 0.4
NO=np.loadtxt("NO_energies.csv",usecols=(0),delimiter=" ") - 0.71
H=np.loadtxt("H_energies.csv",usecols=(0),delimiter=" ") - 0.1



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
    
    im=ax.imshow(prob_density.T,norm=LogNorm(vmin=1e-8,vmax=1.0),cmap=cmap,interpolation='gaussian',origin='lower',extent=[xedges[0], xedges[-1],yedges[0],yedges[-1]],aspect="auto")
    cbar=plt.colorbar(im,ticks=[1e-8,1e-5,1e-4,0.001,0.01,0.1,1])
    cbar.ax.set_yticklabels(["0","$10^{-5}$","$10^{-4}$","$10^{-3}$","$10^{-2}$","$10^{-1}$","1"]) 
    cbar.ax.set_ylabel("Probability Density")
    if return_hists:
        return fig,ax,(hist_x,xedges),(hist_y,yedges), prob_density
    else:
        return fig,ax

#CO vs H plot    
fig,ax=probability_density_plot(H, CO)
ax.set_xlabel("\u0394$E_{pred}^{*H}$ relative to \u0394$E_{Cu(111)}^{*H}$ [eV]")
ax.set_ylabel("\u0394$E_{pred}^{*CO}$ relative to \u0394$E_{Cu(111)}^{*CO}$ [eV]")
plt.title("$Ag_{20}Au_{20}Cu_{20}Pd_{20}Pt_{20}$")
ax.scatter(H_energies,CO_energies)
for (x,y,s) in zip(H_energies,CO_energies,metals):
    ax.text(x,y,s)


#NO vs H plot    
fig,ax=probability_density_plot(H, NO)
ax.set_xlabel("\u0394$E_{pred}^{*H}$ relative to \u0394$E_{Cu(111)}^{*H}$ [eV]")
ax.set_ylabel("\u0394$E_{pred}^{*NO}$ relative to \u0394$E_{Cu(111)}^{*NO}$ [eV]")
plt.title("$Ag_{20}Au_{20}Cu_{20}Pd_{20}Pt_{20}$")
ax.scatter(H_energies,NO_energies)
for (x,y,s) in zip(H_energies,NO_energies,metals):
    ax.text(x,y,s)

    
#NO vs CO
fig,ax,(hist_x,xedges),(hist_y,yedges), prob_density=probability_density_plot(CO, NO,return_hists=True)
ax.set_ylabel("\u0394$E_{pred}^{*NO}$ relative to \u0394$E_{Cu(111)}^{*NO}$ [eV]")
ax.set_xlabel("\u0394$E_{pred}^{*CO}$ relative to \u0394$E_{Cu(111)}^{*CO}$ [eV]")
plt.title("$Ag_{20}Au_{20}Cu_{20}Pd_{20}Pt_{20}$")
ax.scatter(CO_energies,NO_energies)
for (x,y,s) in zip(CO_energies,NO_energies,metals):
    ax.text(x+0.01,y+0.01,s)




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
prob=(sum(mask_x(CO))/10000) * (sum(mask_y(NO))/10000)
ax.text((-0.4-1.1)/2,(-1.3-0.71)/2,f"Urea \nProbability: {prob:1.3f}",c="b",alpha=0.8,ha="center",va="center")
#ax.text((-0.4-1.1)/2,-0.82,f"Prob. density:{prob_density_urea:1.3f}",c="tab:blue",ha="center",va="center")


