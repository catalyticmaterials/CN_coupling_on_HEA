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
#P_NO = 1
method = 'mc'

grid_mf = get_molar_fractions(0.05,3)

grid_mf_quinary = np.hstack((np.zeros((len(grid_mf),1)),grid_mf,np.zeros((len(grid_mf),1))))


results_1 = np.empty((0,4))
results_01 = np.empty((0,4))
for composition in tqdm(grid_mf_quinary):
    # n_H, n_NO_NH3, n_NO_CN, n_pairs = count_selectivity(composition,P_CO,P_NO, metals, method,n=5)
    results_1 = np.vstack((results_1,count_selectivity(composition,P_CO,1, metals, method)))
    results_01 = np.vstack((results_01,count_selectivity(composition,P_CO,0.1, metals, method)))

None_array = np.full((len(results_1),1),'')
data = np.hstack((grid_mf,None_array,results_1,None_array,results_01))

np.savetxt('AuCuPd_mc_selectivity.csv',data,delimiter=',',fmt="%s",
           header = 'Au,Cu,Pd,P_NO=1,H,NO_NH3,NO_CN,CO_NO_pairs,P_NO=0.1,H,NO_NH3,NO_CN,CO_NO_pairs')
