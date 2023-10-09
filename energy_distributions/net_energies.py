import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatch
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from density_plot_functions import probability_density_plot
import sys
sys.path.append('..')
from scripts.methods import get_sites
from scripts import metals, G_corr


plt.rc('font', size=8)
plt.rc('legend', fontsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('xtick', labelsize=8) 

C_color = np.array([80,80,80])/255
N_color = np.array([48,80,248])/255
H_color = np.array([179,227,245])/255

for method in ['eq','dyn']:
    np.random.seed(1)
    composition = np.array([0.2,0.2,0.2,0.2,0.2])
    CO_NO_energy_pairs, CO_ads, NO_ads, H_ads, CO_energies, NO_energies, H_energies, CO_bool, NO_bool, H_bool = get_sites(composition,P_CO=1,P_NO=1, metals=metals, method=method, eU=0, n=100, return_ads_energies=True)

    fig, ax = plt.subplots(figsize=(3.4,2))
    Emin = np.min([CO_energies,NO_energies,H_energies])
    Emax = np.max(np.concatenate([CO_energies,NO_energies,H_energies]))

    range = (Emin,Emax)
    bins=100
    ax.hist(CO_energies,histtype='step',label='CO',color=C_color,range=range,bins=bins,zorder=2)
    ax.hist(NO_energies,histtype='step',label='NO',color=N_color,range=range,bins=bins,zorder=1)
    # ax.hist(H_energies[H_energies<10],histtype='step',label='H',color=H_color,range=range,bins=bins,zorder=0)
    CO_counts, bins, patches = ax.hist(CO_ads,label='*CO',alpha=1,range=range,bins=bins,color=C_color,zorder=2)
    ax.hist(NO_ads,label='*NO',alpha=0.6,range=range,bins=bins,color=N_color,zorder=1)
    # ax.hist(H_ads,label='*H',alpha=0.4,range=range,bins=bins,color=H_color,zorder=0)
    # ax.set_xlabel('Adsorption Gibbs Free Energies [eV]')
    ax.set_xlabel('$\Delta G_{ads}$ [eV]',fontsize=9)
    ax.set_ylabel('Number of Sites',fontsize=9)

    
    ax.set_ylim(None,380)
    ax.set_xlim(-1.5,0.1)
    
    h,l = ax.get_legend_handles_labels()
    ax.legend(loc=2,handles=h[2:],labels=l[2:])


    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(50))



    # xbins = (bins[1:] + bins[:-1])/2
    # arrow_mask = (xbins < -1.0) * (CO_counts<=5) * (CO_counts>0)
    
    # x_arrow = xbins[arrow_mask]
    # y_arrow = CO_counts[arrow_mask]
    
    # x_start = -1.25
    # y_start = 25

    # prop = dict(arrowstyle="-|>,head_width=0.05,head_length=0.2", shrinkA=0,shrinkB=0,color='k')
    # for x_end,y_end in zip(x_arrow,y_arrow):
    #     plt.annotate('',xy=(x_end,y_end),xytext=(x_end,y_start),arrowprops=prop)
        
    
    # ax.add_artist(mpatch.Rectangle((min(x_arrow)-0.005,30),max(x_arrow)-min(x_arrow)+0.01,22,fill=False,edgecolor='black',linewidth=0.6,linestyle='--'))
    # ax.text((max(x_arrow)+min(x_arrow))/2, 30,'5 or less adsorbed CO',ha='center',va='bottom',fontsize=7)



    if method == 'dyn':
        axins = ax.inset_axes([0.03, 0.46, 0.2, 0.2])
        # axins = zoomed_inset_axes(ax, 5, loc=1)
        axins.hist(CO_ads[CO_ads<=-1],range=range,bins=bins,color=C_color)
        axins.set_xlim([-1.4,-1.0])
        axins.set_ylim([0,5.5])
        axins.set_yticks([])
        axins.xaxis.set_major_locator(MultipleLocator(0.5))
        axins.xaxis.set_minor_locator(MultipleLocator(0.1))
        #axins.yaxis.set_major_locator(MultipleLocator(5))
        axins.yaxis.set_minor_locator(MultipleLocator(1))


        #axins.set_xticklabels([])
        #axins.set_yticklabels([])
        # ax.indicate_inset_zoom(axins, edgecolor="black")
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="black",alpha=0.3,linestyle='--')


    plt.tight_layout()
    plt.savefig(f'adsorption_dist_{method}.png',dpi=600)
    # plt.savefig('test.png',dpi=600)
 
    # plt.show()
    # plt.close()


    #Correct to adsorption energies
    NO_energies-=G_corr['NO']
    CO_energies-=G_corr['CO']
    H_energies-=G_corr['H']

    #Correct to adsorption energies
    NO_ads-=G_corr['NO']
    CO_ads-=G_corr['CO']
    CO_NO_energy_pairs[:,0] -= G_corr['CO']
    CO_NO_energy_pairs[:,1] -= G_corr['NO']



    H_bool_pad = np.pad(H_bool,pad_width=((1,1),(1,1)),mode='wrap')

    CO_block_mask = H_bool_pad[1:-1,1:-1] + H_bool_pad[:-2,1:-1] + H_bool_pad[1:-1,:-2]

    NO_energies = NO_energies[np.invert(H_bool).flatten()]
    CO_energies = CO_energies[np.invert(CO_block_mask).flatten()]




    # fig,ax,(hist_x,xedges),(hist_y,yedges), prob_density = probability_density_plot(CO_energies,NO_energies,return_hists=True)
    

    
    fig,ax = plt.subplots(figsize=(4,4))
    ax.set_ylim(-1.455,-0.123)
    ax.set_xlim(-1.82,-0.0265)
    

    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    

    fig,ax,(hist_x,xedges),(hist_y,yedges), prob_density = probability_density_plot(CO_energies,NO_energies,return_hists=True,figure=(fig,ax),vmax=80)


    ax.scatter(*CO_NO_energy_pairs.T,marker='.',c='b',edgecolors='k',label=f'CO-NO pairs: {len(CO_NO_energy_pairs)}',s=14,linewidths=0.6)
    ax.legend(loc=2, markerscale=2.,framealpha=0.4)

    a = 1.88
    b = - 1.24




    ax.vlines(-G_corr['CO'], ylim[0],ylim[1],color='k',ls='--',alpha=0.6,lw=1)
    ax.hlines(-G_corr['NO'], xlim[0],xlim[1],color='k',ls='--',alpha=0.6,lw=1)
    ax.hlines(a*(-0.1)+b, xlim[0],xlim[1],color='b',ls='--',lw=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


    ax.set_xlabel("\u0394$E_{pred}^{*CO}$ [eV]")
    ax.set_ylabel("\u0394$E_{pred}^{*NO}$ [eV]")


    ax.xaxis.set_major_locator(MultipleLocator(0.4))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))



  


    # x_centers = (xedges[1:] + xedges[:-1])/2
    # y_centers = (yedges[1:] + yedges[:-1])/2

    # x_centers_id=np.where(x_centers<=-0.4)[0]
    # y_centers_id=np.where(y_centers<=-0.71)[0]

    #prob2 = np.sum(prob_density.T[y_centers_id[0]:y_centers_id[-1]+1,x_centers_id[0]:x_centers_id[-1]+1]) * (x_centers[x_centers_id[1]]-x_centers[x_centers_id[0]]) * (y_centers[y_centers_id[1]]-y_centers[y_centers_id[0]]) 

    #print(prob2)
    # alloy_sub_list = [metals[i] + '$_{' + str(int(composition[i]*100)) + '}$' for i in range(len(metals)) if composition[i]>0]
    # alloy_sub = ''.join(alloy_sub_list)
    # props = dict(boxstyle='round', facecolor='white', alpha=0.3)
    # ax.text(xlim[0]+0.05,ylim[1]-0.05,alloy_sub,ha='left',va='top',color='k',bbox=props)

    # ax.text(xlim[0]+0.02,-0.73,f'$P_{{area}}$ = {probability:.4f}',ha='left',va='top',color='tab:blue')
    # ax.fill_between([xlim[0],-0.4],ylim[0],-0.71,alpha=0.3,hatch='///',color='none',linewidth=0,edgecolor='tab:blue')
    

    #ax.set_aspect('equal')
    #plt.tight_layout()
    #plt.show()
    # plt.savefig('test.png',dpi=400)
    # break


    fig.subplots_adjust(bottom=0.2,left=0.2,right=0.8,top=0.8)


    plt.savefig(f'Energy_dist_{method}.png',dpi=600,bbox_inches='tight')
    # plt.show()
