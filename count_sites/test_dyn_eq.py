import numpy as np
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from methods import block_indices, adsorb_CO,adsorb_H,adsorb_NO
import sys
sys.path.append("..")
from shared_params.surface import predict_energies, initiate_surface


#composition = np.array([0.2,0.2,0.2,0.2,0.2])
composition = np.array([0.0,0.0,1.0,0.0,0.0])

P_CO=1
P_NO=0.1
max_iterations=int(5.5e+5)
n=100
eU=0
H_UPD=True

# Define Boltzmann's constant
#kB = 1.380649e-4 / 1.602176634  # eV K-1 (exact)
kB=8.617333262 * 1e-5 #eV/K
kBT = kB*300 # eV

metals = ['Ag','Au', 'Cu', 'Pd','Pt']




t_start = time()
#Initiate surface
surface = initiate_surface(composition,metals,size=(n,n))

#Predict energies of every site
CO_energies,CO_site_ids=predict_energies(surface, "CO", metals)
NO_energies,NO_site_ids=predict_energies(surface, "NO_fcc", metals)
H_energies,H_site_ids = predict_energies(surface,"H",metals)


#Adjust free energy by chemical potential from preassure
CO_energies += kBT * np.log(1/P_CO)
NO_energies += kBT * np.log(1/P_NO)

#Turn into grids
CO_energy_grid = CO_energies.reshape(n,n)
NO_energy_grid = NO_energies.reshape(n,n)
H_energy_grid = H_energies.reshape(n,n)

CO_data = np.hstack((CO_energies.reshape(-1,1),CO_site_ids))
NO_data = np.hstack((NO_energies.reshape(-1,1),NO_site_ids))
H_data = np.hstack((H_energies.reshape(-1,1),H_site_ids))

#Only keep negative energies
CO_data = CO_data[CO_data[:,0]<0]
NO_data = NO_data[NO_data[:,0]<0]


#Coverage masks
CO_coverage_mask = np.full((n, n),None)
NO_coverage_mask = np.full((n, n),None)
H_coverage_mask = np.full((n, n),None)



if H_UPD:
    #fill surface with H where adsorption energy is negative
    for (energy, i, j) in H_data:
        i,j=int(i),int(j)
        if energy<=(-eU):
            H_coverage_mask,CO_coverage_mask,NO_coverage_mask = adsorb_H(i,j,H_coverage_mask, CO_coverage_mask, NO_coverage_mask,n)


H_coverage_bool = np.asanyarray(H_coverage_mask,dtype=bool)

P = P_CO + P_NO

p_CO = P_CO/P
p_NO=P_NO/P

energies = []
ite = []
energy1=0
nads=0
d_nads = []
break_bool=True

for iteration in range(max_iterations):

    r = np.random.rand()
    if r<= p_CO:

        ind = np.random.choice(len(CO_data))#,p=p)
        energy, i, j = CO_data[ind]
        i,j = int(i),int(j)
        #if CO_coverage_mask[i,j] and iteration%1000!=0: continue
        #CO_data = np.delete(CO_data, ind,axis=0)
        block_vectors = block_indices(i, j, "top",n)
        #NO_blocked = np.any([NO_coverage_mask[ib,jb] for (ib,jb) in block_vectors])
        if H_UPD:
            H_blocked = np.any([H_coverage_mask[ib,jb] for (ib,jb) in block_vectors])
            if H_blocked and CO_coverage_mask[i,j] is None: 
                print("CO",i,j)
                break
        else: H_blocked=False
        if H_blocked: #continue
            pass

        elif CO_coverage_mask[i,j] is None:
            CO_coverage_mask,NO_coverage_mask=adsorb_CO(i, j, CO_coverage_mask, NO_coverage_mask,n)
            nads+=1
       
        elif CO_coverage_mask[i,j]==False:
            
            E=0
            NO_blocking_ind = np.empty((0,2),dtype="int")
            for (i_b,j_b) in block_vectors:
                if NO_coverage_mask[i_b,j_b]:
                    E += NO_energy_grid[i_b,j_b]
                    NO_blocking_ind = np.vstack((NO_blocking_ind,np.array([[i_b,j_b]])))
            
            Delta_E = CO_energy_grid[i,j] - E
            if Delta_E<0 or np.random.rand() <= np.exp(-Delta_E/kBT):
                #unblock 
                for i_b,j_b in NO_blocking_ind:
                    i_ub,j_ub = block_indices(i_b, j_b, "fcc",n).T
                    not_H_blocked = np.array([np.any(H_coverage_bool[block_indices(ib_, jb_, "top",n).T])==False for ib_,jb_ in zip(i_ub,j_ub)])
                    CO_coverage_mask[i_ub,j_ub][not_H_blocked]=None
                CO_coverage_mask,NO_coverage_mask=adsorb_CO(i, j, CO_coverage_mask, NO_coverage_mask,n)
                #if Delta_E>0: print(Delta_E,np.exp(-Delta_E/kBT))
    else:        
        ind = np.random.choice(len(NO_data))#,p=p)
        energy, i, j = NO_data[ind]
        i,j = int(i),int(j)
        #if NO_coverage_mask[i,j] and iteration%1000!=0: continue
        #NO_data = np.delete(NO_data, ind,axis=0)
        block_vectors = block_indices(i, j, "fcc",n)
        #CO_blocked = np.any([CO_coverage_mask[ib,jb] for (ib,jb) in block_vectors])
        if H_UPD:
            H_blocked = H_coverage_mask[i,j]==True
            if H_blocked and NO_coverage_mask[i,j] is None: print("error")
        else: H_blocked = False
        
        if H_blocked: pass #continue


        elif NO_coverage_mask[i,j] is None:
            CO_coverage_mask,NO_coverage_mask=adsorb_NO(i, j, CO_coverage_mask, NO_coverage_mask,n)
            nads+=1
        #elif NO_coverage_mask[i,j]==False and CO_blocked:
        elif NO_coverage_mask[i,j]==False:
                
                E=0
                CO_blocking_ind = np.empty((0,2),dtype="int")
                for (i_b,j_b) in block_vectors:
                    if CO_coverage_mask[i_b,j_b]:
                        E += CO_energy_grid[i_b,j_b]
                        CO_blocking_ind = np.vstack((CO_blocking_ind,np.array([[i_b,j_b]])))
                        
                Delta_E = NO_energy_grid[i,j] - E
                if Delta_E<0 or np.random.rand() <= np.exp(-Delta_E/kBT):
                    #unblock
                    for i_b,j_b in CO_blocking_ind:
                        i_ub,j_ub = block_indices(i_b, j_b, "top",n).T
                        not_H_blocked = H_coverage_bool[i_ub,j_ub]==False
                        NO_coverage_mask[i_ub,j_ub][not_H_blocked]=None
                    CO_coverage_mask,NO_coverage_mask=adsorb_NO(i, j, CO_coverage_mask, NO_coverage_mask,n)
    
    
    
    if (iteration+1)%1000==0:
        CO_ads = CO_energy_grid[np.asanyarray(CO_coverage_mask,dtype=bool)]
        NO_ads = NO_energy_grid[np.asanyarray(NO_coverage_mask,dtype=bool)]
        H_ads = H_energy_grid[np.asanyarray(H_coverage_mask,dtype=bool)]
        energy2 = np.sum(CO_ads)+np.sum(NO_ads)+np.sum(H_ads)
        energies.append(energy2)
        ite.append(iteration+1)
        d_nads.append(nads/1000)
        #nads=0
        #Delta_energy = abs(energy2-energy1)
        #if Delta_energy < 0.001: 
        #    break
        #else: energy1 = energy2
        if nads==0 and break_bool:
            #break
            break_iteration = iteration+1
            break_bool=False
        else: nads=0
    
    
CO_coverage_bool = np.asanyarray(CO_coverage_mask,dtype=bool)
NO_coverage_bool = np.asanyarray(NO_coverage_mask,dtype=bool)



#Pad grids
NO_coverage_mask_pad = np.pad(NO_coverage_bool,pad_width=((1,1),(1,1)),mode="wrap")

pairs1_bool=CO_coverage_bool * NO_coverage_mask_pad[:-2,:-2]
pairs2_bool=CO_coverage_bool * NO_coverage_mask_pad[2:,:-2]
pairs3_bool=CO_coverage_bool * NO_coverage_mask_pad[:-2,2:]

n_pairs = np.sum(pairs1_bool) + np.sum(pairs2_bool) + np.sum(pairs3_bool)
    
    
surface_atoms = np.multiply(*surface.shape[:2])                 
 
active_sites = n_pairs/surface_atoms

print(str(composition),"fractional active sites:",active_sites,"evaluation time (s):",time()-t_start)


fig,ax = plt.subplots(dpi=400)
l1,=ax.plot(ite,energies,c="tab:blue",label="Total ads. energy")
ax2=ax.twinx()
l2,=ax2.plot(ite,d_nads,c="tab:orange",label="Adsorptions pr. iterations")

ax.legend(handles=[l1,l2])

ax.set_xlabel("Number of iterations")
ax.set_ylabel("Energy (eV)")
ax2.set_ylabel("Direct adsorption rate")
ylim=ax.get_ylim()
ax.plot([break_iteration]*2,ylim,c="k",ls="--")
ax.set_xlim(None,5e+5)

#text="Ag$_{20}$Au$_{20}$Cu$_{20}$Pd$_{20}$Pt$_{20}$\nP$_{CO}$ = 1, P$_{NO}$ = 0.5"
#ax.text(2.5e+5,-1250 , s=text)
text = "Cu, P$_{CO}$ = 1, P$_{NO}$ = 0.1"
ax.text(2.5e+5,-700 , s=text)