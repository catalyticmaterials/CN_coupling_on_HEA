import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append('..')
from scripts import metals
from scripts.methods import count_selectivity
from scripts import get_molar_fractions
from scripts.plotting_ternary_functions import make_plot, molar_fractions_to_cartesians


np.random.seed(42)
#composition = np.array([0.2,0.2,0.2,0.2,0.2])
P_CO = 1
P_NO = 1
method = 'mc'


grid_mf = get_molar_fractions(0.1,3)

grid_mf_quinary = np.hstack((np.zeros((len(grid_mf),1)),grid_mf,np.zeros((len(grid_mf),1))))

results = np.empty((0,4))
for composition in tqdm(grid_mf_quinary):
    # n_H, n_NO_NH3, n_NO_CN, n_pairs = count_selectivity(composition,P_CO,P_NO, metals, method,n=5)
    results = np.vstack((results,count_selectivity(composition,P_CO,P_NO, metals, method)))


data = np.hstack((grid_mf,results))

np.savetxt('AuCuPd_selectivity.csv',data,delimiter=',',fmt=['%.2f','%.2f','%.2f','%.4f','%.4f','%.4f','%.4f'])

grid = molar_fractions_to_cartesians(grid_mf)


fig,ax = make_plot(grid,results[:,0],metals[1:4])
plt.savefig('test_H.png',dpi=400)

fig,ax = make_plot(grid,results[:,1],metals[1:4],colormap='plasma_r')
plt.savefig('test_NH3.png',dpi=400)

fig,ax = make_plot(grid,results[:,3],metals[1:4],colormap='summer_r')
plt.savefig('test_pairs.png',dpi=400)





