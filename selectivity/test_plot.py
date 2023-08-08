import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from scripts import metals
from scripts.methods import count_selectivity
from scripts import get_molar_fractions
from scripts.plotting_ternary_functions import make_plot, molar_fractions_to_cartesians


results = np.loadtxt('AuCuPd_selectivity.csv',delimiter=',',usecols=(3,4,5,6))


grid_mf = get_molar_fractions(0.1,3)
grid = molar_fractions_to_cartesians(grid_mf)

fig,ax = make_plot(grid,results[:,0],metals[1:4],colormap='plasma_r',minval=0.0,colorbar=True)
plt.tight_layout()
plt.savefig('test_H.png',dpi=400)

fig,ax = make_plot(grid,results[:,1],metals[1:4],colormap='summer_r',minval=0.0,colorbar=True)
plt.tight_layout()
plt.savefig('test_NH3.png',dpi=400)

NO_conversion = results[:,2]/(results[:,1]+results[:,2])
NO_conversion[np.isnan(NO_conversion)] = 0
fig,ax = make_plot(grid,NO_conversion,metals[1:4],colormap='cividis',minval=0.0,colorbar=True)
plt.tight_layout()
plt.savefig('test_CNvsNH3.png',dpi=400)

fig,ax = make_plot(grid,results[:,3],metals[1:4],colorbar=True)
plt.tight_layout()
plt.savefig('test_pairs.png',dpi=400)