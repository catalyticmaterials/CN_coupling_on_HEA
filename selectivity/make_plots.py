import numpy as np
import matplotlib.pyplot as plt

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


    fig,ax = make_plot(grid,H,metals,colormap='plasma_r',minval=0.0,colorbar=True)
    plt.tight_layout()
    plt.savefig(f'AuCuPd_H_PNO{P_NO}.png',dpi=400)

    fig,ax = make_plot(grid,NO_NH3,metals,colormap='summer_r',minval=0.0,colorbar=True)
    plt.tight_layout()
    plt.savefig(f'AuCuPd_NH3_PNO{P_NO}.png',dpi=400)

    NO_conversion = NO_CN/(NO_NH3+NO_CN)
    NO_conversion[np.isnan(NO_conversion)] = 0
    fig,ax = make_plot(grid,NO_conversion,metals,colormap='cividis',minval=0.0,colorbar=True)
    plt.tight_layout()
    plt.savefig(f'AuCuPd_CNvsNH3_PNO{P_NO}.png',dpi=400)

    fig,ax = make_plot(grid,pairs,metals,colorbar=True)
    plt.tight_layout()
    plt.savefig(f'AuCuPd_pairs__PNO{P_NO}.png',dpi=400)