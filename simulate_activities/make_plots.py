import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from scripts.compositionspace_functions import molar_fractions_to_cartesians
import scripts.plotting_ternary_functions as pt


data = np.loadtxt("AuCuPd_grid.csv",delimiter=",",skiprows=1)

grid_mf, rate1,rate01 = data[:,:3],data[:,3],data[:,4]


grid = molar_fractions_to_cartesians(grid_mf)



fig, ax = pt.make_plot(grid.T, rate1, ["Au","Cu","Pd"],colorbar=True)
plt.savefig('AuCuPd_PNO1.png')

fig, ax = pt.make_plot(grid.T, rate01, ["Au","Cu","Pd"],colorbar=True)
plt.savefig('AuCuPd_PNO01.png')