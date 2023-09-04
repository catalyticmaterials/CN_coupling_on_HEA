import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import sys
sys.path.append('..')
from scripts import metals
from scripts.methods import count_selectivity
from scripts import get_molar_fractions
from scripts.plotting_ternary_functions import make_plot, molar_fractions_to_cartesians


grid_mf = get_molar_fractions(0.05,3)
grid = molar_fractions_to_cartesians(grid_mf)

metals = metals[1:4]

for P_NO, usecols in zip(['1','01'],[(4,5,6,7),(9,10,11,12)]):

    H, NO_NH3, NO_CN, pairs = np.loadtxt('AuCuPd_mc_selectivity.csv',delimiter=',',usecols=usecols).T


    # fig,ax = make_plot(grid,H,metals,colormap='plasma_r',minval=0.0,colorbar=True)
    # plt.tight_layout()
    # plt.savefig(f'AuCuPd_H_PNO{P_NO}.png',dpi=400)

    # fig,ax = make_plot(grid,NO_NH3,metals,colormap='summer_r',minval=0.0,colorbar=True)
    # plt.tight_layout()
    # plt.savefig(f'AuCuPd_NH3_PNO{P_NO}.png',dpi=400)

    # fig,ax = make_plot(grid,NO_CN,metals,colormap='summer',minval=0.0,colorbar=True)
    # plt.tight_layout()
    # plt.savefig(f'AuCuPd_CN_PNO{P_NO}.png',dpi=400)

    NO_conversion = NO_CN/(NO_NH3+NO_CN+H)
    # NO_conversion[np.isnan(NO_conversion)] = 0
    # fig,ax = make_plot(grid,NO_conversion,metals,colormap='cividis',minval=0.0,colorbar=True)
    # plt.tight_layout()
    # plt.savefig(f'AuCuPd_CNvsNH3_PNO{P_NO}.png',dpi=400)

    # fig,ax = make_plot(grid,pairs,metals,colorbar=True)
    # plt.tight_layout()
    # plt.savefig(f'AuCuPd_pairs_PNO{P_NO}.png',dpi=400)


    fig,ax = plt.subplots(figsize=(4,4))
    ax.scatter(NO_conversion,pairs,c='k',marker='.',label='Compositions')
    ax.legend(loc=2)
    ax.set_ylabel('CO-NO pairs per surface atom')
    ax.set_xlabel('CN conversion')
    ax.text(0.02,0.82,f'P$_{{NO}}$ = {P_NO}',transform=ax.transAxes)

    if P_NO=='1':
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))
        # ax.xaxis.set_minor_locator(MultipleLocator(0.01))
        ax.set_xlim(None,0.2)
    elif P_NO=='01':
        # ax.yaxis.set_minor_locator(MultipleLocator(0.01))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    


    # plt.tight_layout()
    plt.subplots_adjust(left=0.2,bottom=0.2)
    plt.savefig(f'pairs_vs_conversion_PNO{P_NO}.png',dpi=600,bbox_inches='tight')