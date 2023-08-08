import numpy as np
import matplotlib.pyplot as plt
from density_plot_functions import probability_density_plot
import sys
sys.path.append('..')
from scripts.methods import get_sites
from scripts import metals

C_color = np.array([100,100,100])/255
N_color = np.array([48,80,248])/255
H_color = np.array([179,227,245])/255

for method in ['eq','dyn']:
    np.random.seed(1)
    composition = np.array([0.2,0.2,0.2,0.2,0.2])
    CO_NO_energy_pairs, CO_ads, NO_ads, H_ads, CO_energies, NO_energies, H_energies = get_sites(composition,P_CO=1,P_NO=1, metals=metals, method=method, eU=0, n=100, return_ads_energies=True)

    fig, ax = plt.subplots(figsize=(8,3))
    Emin = np.min([CO_energies,NO_energies,H_energies])
    Emax = np.max(np.concatenate([CO_energies,NO_energies,H_energies[H_energies<10]]))

    range = (Emin,Emax)
    bins=100
    plt.hist(CO_energies,histtype='step',label='CO',color=C_color,range=range,bins=bins,zorder=2)
    plt.hist(NO_energies,histtype='step',label='NO',color=N_color,range=range,bins=bins,zorder=1)
    plt.hist(H_energies[H_energies<10],histtype='step',label='H',color=H_color,range=range,bins=bins,zorder=0)
    plt.hist(CO_ads,label='*CO',alpha=0.7,range=range,bins=bins,color=C_color,zorder=2)
    plt.hist(NO_ads,label='*NO',alpha=0.7,range=range,bins=bins,color=N_color,zorder=1)
    plt.hist(H_ads,label='*H',alpha=0.7,range=range,bins=bins,color=H_color,zorder=0)
    plt.xlabel('Adsorption Gibbs Free Energies [eV]')
    plt.ylabel('Number of Sites')
    plt.ylim(None,550)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'adsorption_dist_{method}.png',dpi=400)
    plt.close()


    #Correct to adsorption energies
    NO_energies-=0.71
    CO_energies-=0.4
    H_energies-=0.1

    #Correct to adsorption energies
    NO_ads-=0.71
    CO_ads-=0.4
    CO_NO_energy_pairs[:,0] -= 0.4
    CO_NO_energy_pairs[:,1] -= 0.71



    fig,ax,(hist_x,xedges),(hist_y,yedges), prob_density = probability_density_plot(CO_energies,NO_energies,return_hists=True)
    ax.scatter(*CO_NO_energy_pairs.T,marker='.',c='b',edgecolors='k',label='CO-NO pairs')

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
    CO_NO_pair_mask_CO = mask_x(CO_NO_energy_pairs[:,0])
    CO_NO_pair_mask_NO = mask_y(CO_NO_energy_pairs[:,1])
    n_pairs_in_area = np.sum(CO_NO_pair_mask_CO * CO_NO_pair_mask_NO)

    ax.text((-0.4-1.1)/2,(-1.3-0.71)/2+0.35,f"pairs in area/surface atom:\n{n_pairs_in_area/10000}",c="b",alpha=0.8,ha="center",va="center")
    ax.legend(loc=2)
    ax.set_ylabel("\u0394$E_{pred}^{*NO}$ [eV]")
    ax.set_xlabel("\u0394$E_{pred}^{*CO}$ [eV]")
    plt.tight_layout()
    #print(prob)
    plt.savefig(f'Energy_dist_{method}.png',dpi=400)
