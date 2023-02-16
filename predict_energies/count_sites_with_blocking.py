import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from shared_params.surface import predict_energies, initiate_surface, characterize_sites
from shared_params.kinetic_model import fractional_urea_rate, urea_rate,urea_conversion

# Define Boltzmann's constant
#kB = 1.380649e-4 / 1.602176634  # eV K-1 (exact)
kB=8.617333262 * 1e-5 #eV/K
kBT = kB*300 # eV


metals = ['Ag','Au', 'Cu', 'Pd','Pt']
composition = np.array([0.2,0.2,0.2,0.2,0.2])
#composition = np.array([0.0,0.0,1.0,0.0,0.0])
#composition = np.array([0.0,0.5,0.5,0.0,0.0])

np.random.seed(1)

#Initiate surface
surface = initiate_surface(composition,metals)


#Predict energies of every site
CO_energies,CO_site_ids=predict_energies(surface, "CO", metals)
NO_energies,NO_site_ids=predict_energies(surface, "NO_fcc", metals)
H_energies,H_site_ids = predict_energies(surface,"H",metals)



P_CO = 1
P_NO=0.017
P_H=1

#Adjust free energy by chemical potential from preassure
CO_energies += kBT * np.log(1/P_CO)
NO_energies += kBT * np.log(1/P_NO)
H_energies += kBT * np.log(1/P_H)
#Turn into grids
CO_energy_grid = CO_energies.reshape(100,100)
NO_energy_grid = NO_energies.reshape(100,100)
H_energy_grid = H_energies.reshape(100,100)



CO_ids=np.ones(len(CO_energies))*1
NO_ids=np.ones(len(NO_energies))*2
#H_ids=np.ones(len(H_energies))*3


CO_data = np.hstack((CO_energies.reshape(-1,1),CO_site_ids,CO_ids.reshape(-1,1)))
NO_data = np.hstack((NO_energies.reshape(-1,1),NO_site_ids,NO_ids.reshape(-1,1)))
H_data = np.hstack((H_energies.reshape(-1,1),H_site_ids))


CO_NO_array = np.vstack((CO_data,NO_data))

#Sort data by lowest energy
CO_NO_array = CO_NO_array[CO_NO_array[:, 0].argsort()]

#Coverage masks
CO_coverage_mask = np.full((100, 100),None)
NO_coverage_mask = np.full((100, 100),None)
H_coverage_mask = np.full((100, 100),None)


# Define relative blocking vectors
block_fcc_vectors = np.array([[-1,0],[0,-1],[0,0]])
block_top_vectors = np.array([[0,0],[0,1],[1,0]])

#fill surface with H where adsorption energy is negative
for (energy, i, j) in H_data:
    i,j=int(i),int(j)
    if energy<=0:
        H_coverage_mask[i,j] = True
        
        #Block sites
        ij_block_vectors = np.array([i,j]) + block_top_vectors
        for (i_b,j_b) in ij_block_vectors:
            if i_b == 100:
                i_b=0
            if j_b == 100:
                j_b=0
            CO_coverage_mask[i_b,j_b] = False
        NO_coverage_mask[i,j] = False


#Fill surface with NO and CO with blocking
for (energy,i,j,idx) in CO_NO_array:
    #if energy>0: break
    i,j,idx = int(i),int(j),int(idx)
    ij_vec = np.array([i,j])
        
    if idx==1:
        if CO_coverage_mask[i,j] is None:
            CO_coverage_mask[i,j] = True
            #CO_energies = np.append(CO_energies,energy)
            
            #Block sites
            ij_block_vectors = ij_vec + block_fcc_vectors
            for (i_b,j_b) in ij_block_vectors:
                if i_b == 100:
                    i_b=0
                if j_b == 100:
                    j_b=0
                NO_coverage_mask[i_b,j_b] = False
                #H_coverage_mask[i_b,j_b] = False
    
    elif idx==2:
        if NO_coverage_mask[i,j] is None:
            NO_coverage_mask[i,j] = True
            #NO_energies = np.append(NO_energies,energy)
            
            #Block sites
            ij_block_vectors = ij_vec + block_top_vectors
            for (i_b,j_b) in ij_block_vectors:
                if i_b == 100:
                    i_b=0
                if j_b == 100:
                    j_b=0
                CO_coverage_mask[i_b,j_b] = False
            #H_coverage_mask[i,j] = False
        



CO_coverage_mask = np.asanyarray(CO_coverage_mask,dtype=bool)
NO_coverage_mask = np.asanyarray(NO_coverage_mask,dtype=bool)
H_coverage_mask = np.asanyarray(H_coverage_mask,dtype=bool)


print("CO coverage:",sum(CO_coverage_mask.flatten()))
print("NO coverage:",sum(NO_coverage_mask.flatten()))
print("H coverage:",sum(H_coverage_mask.flatten()))


CO_ads = CO_energy_grid[CO_coverage_mask]
NO_ads = NO_energy_grid[NO_coverage_mask]
H_ads = H_energy_grid[H_coverage_mask]

plt.figure()
plt.hist(CO_ads,bins=100,histtype="step",label="CO")
plt.hist(NO_ads,bins=100,histtype="step",label="NO")
plt.hist(H_ads,bins=100,histtype="step",label="H")
plt.legend()


#Get site ids of where CO has adsorbed
CO_ads_sites = np.array(np.where(CO_coverage_mask==True)).T

CO_NO_energy_pair = np.empty((0,2))


#Pad grids
NO_coverage_mask_pad = np.pad(NO_coverage_mask,pad_width=((0,1),(0,1)),mode="wrap")
NO_energy_grid_pad = np.pad(NO_energy_grid,pad_width=((0,1),(0,1)),mode="wrap")


#Get CO-NO pairs of catalytic neighboring sites
for (i,j) in CO_ads_sites:
    if NO_coverage_mask_pad[i-1,j-1] == True:
        E_CO = CO_energy_grid[i,j]#CO_energies[i*100+j]
        E_NO = NO_energy_grid_pad[i-1,j-1] #NO_energies[(i-1)*100+(j-1)]
        CO_NO_energy_pair = np.vstack((CO_NO_energy_pair,np.array([[E_CO,E_NO]])))
    if NO_coverage_mask_pad[i-1,j+1] == True:
        E_CO = CO_energy_grid[i,j]#CO_energies[i*100+j]
        E_NO = NO_energy_grid_pad[i-1,j+1] #NO_energies[(i-1)*100+(j+1)]
        CO_NO_energy_pair = np.vstack((CO_NO_energy_pair,np.array([[E_CO,E_NO]])))
    if NO_coverage_mask_pad[i+1,j-1] == True:
        E_CO = CO_energy_grid[i,j] #CO_energies[i*100+j]
        E_NO = NO_energy_grid_pad[i+1,j-1] #NO_energies[(i+1)*100+(j-1)]
        CO_NO_energy_pair = np.vstack((CO_NO_energy_pair,np.array([[E_CO,E_NO]])))
        
#Get all NO sites
NO_ads_sites = np.array(np.where(NO_coverage_mask==True)).T


surface_atoms = np.multiply(*surface.shape[:2])
n_CO_NO_pairs = len(CO_NO_energy_pair)
n_sites = surface_atoms*3

print(n_CO_NO_pairs/n_sites)