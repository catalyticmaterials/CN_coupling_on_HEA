import numpy as np
from tqdm import tqdm
import sys
sys.path.append("..")
from simulate_activity_functions import simulate_activity
from scripts.compositionspace_functions import get_molar_fractions, molar_fractions_to_cartesians
import scripts.plotting_ternary_functions as pt

grid_mf = get_molar_fractions(0.05, 3)

grid_mf = np.hstack((grid_mf,np.zeros((len(grid_mf),2))))

rate1=np.empty(0)
rate05=np.empty(0)
for composition in tqdm(grid_mf):
    rate1 = np.append(rate1,simulate_activity(composition, P_CO=1.0, P_NO=1.0))
    rate05 = np.append(rate05,simulate_activity(composition, P_CO=1.0, P_NO=0.5))
data = np.hstack((grid_mf[:,:3],rate1.reshape(-1,1),rate05.reshape(-1,1)))


np.savetxt("AgAuCu_grid.csv",data,delimiter=",",header="Ag,Au,Cu,active sites (P_NO=1),active sites (P_NO=0.5)",comments="",fmt=["%.2f","%.2f","%.2f","%.6f","%.6f"])

grid = molar_fractions_to_cartesians(grid_mf[:,:3])

pt.make_plot(grid.T, rate1, ["Ag","Au","Cu"],colorbar=True)
pt.make_plot(grid.T, rate05, ["Ag","Au","Cu"],colorbar=True)


