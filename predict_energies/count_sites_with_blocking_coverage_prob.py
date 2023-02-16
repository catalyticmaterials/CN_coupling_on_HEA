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
#composition = np.array([0.2,0.2,0.2,0.2,0.2])
composition = np.array([0.0,0.0,1.0,0.0,0.0])
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
H_ids=np.ones(len(H_energies))*3


CO_data = np.hstack((CO_energies.reshape(-1,1),CO_site_ids,CO_ids.reshape(-1,1)))
NO_data = np.hstack((NO_energies.reshape(-1,1),NO_site_ids,NO_ids.reshape(-1,1)))
H_data = np.hstack((H_energies.reshape(-1,1),H_site_ids,H_ids.reshape(-1,1)))


data_array = np.vstack((CO_data,NO_data,H_data))

#Sort data by lowest energy
data_array = data_array[data_array[:, 0].argsort()]

#Coverage masks
CO_coverage_mask = np.full((100, 100),None)
NO_coverage_mask = np.full((100, 100),None)
H_coverage_mask = np.full((100, 100),None)

#Calculate reaction constant for adsorption
K_CO = np.exp(-CO_energy_grid/kBT)
K_NO = np.exp(-NO_energy_grid/kBT)
K_H = np.exp(-H_energy_grid/kBT)

#Calculate coverage for each site to use as probability for adsorption
#CO_coverage = K_CO*P_CO/(1 + K_CO*P_CO)
#NO_coverage = K_NO*P_NO/(1 + K_NO*P_NO + K_H*P_H)
#H_coverage = K_H*P_H/(1 + K_NO*P_NO + K_H*P_H)
empty_coverage = 1/(1 + K_CO*P_CO + K_NO*P_NO + K_H*P_H)
CO_coverage = K_CO*P_CO * empty_coverage
NO_coverage = K_NO*P_NO * empty_coverage
H_coverage = K_H*P_H * empty_coverage

# Define relative blocking vectors
block_fcc_vectors = np.array([[-1,0],[0,-1],[0,0]])
block_top_vectors = np.array([[0,0],[0,1],[1,0]])

#Fill surface with blocking
while np.any(np.concatenate((CO_coverage_mask,NO_coverage_mask,H_coverage_mask))==None):
    #Draw random numbers between 0 and 1 for the grid
    random_numbers = np.random.random(size=(100,100))
    for (energy,i,j,idx) in data_array:
        #if energy>0: break
        i,j,idx = int(i),int(j),int(idx)
        ij_vec = np.array([i,j])
        r = random_numbers[i,j]
            
        if idx==1 and r<=CO_coverage[i,j]:
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
                    H_coverage_mask[i_b,j_b] = False
        
        elif idx==2 and r<=NO_coverage[i,j]:
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
                H_coverage_mask[i,j] = False
        
        elif idx==3 and r<= H_coverage[i,j]:
            if H_coverage_mask[i,j] is None:
                H_coverage_mask[i,j] = True
                #NO_energies = np.append(NO_energies,energy)
                
                #Block sites
                ij_block_vectors = ij_vec + block_top_vectors
                for (i_b,j_b) in ij_block_vectors:
                    if i_b == 100:
                        i_b=0
                    if j_b == 100:
                        j_b=0
                    CO_coverage_mask[i_b,j_b] = False
                NO_coverage_mask[i,j] = False



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



print(len(CO_NO_energy_pair))