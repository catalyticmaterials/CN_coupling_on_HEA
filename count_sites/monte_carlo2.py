import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from methods import adsorb_CO,adsorb_H,adsorb_NO
import sys
sys.path.append("..")
from shared_params.surface import predict_energies, initiate_surface, characterize_sites
from shared_params.kinetic_model import fractional_urea_rate, urea_rate,urea_conversion


# Define Boltzmann's constant
#kB = 1.380649e-4 / 1.602176634  # eV K-1 (exact)
kB=8.617333262 * 1e-5 #eV/K
kBT = kB*300 # eV
#kBT = kB*0.0000000001 # eV
#kBT=kB*10000000000000

metals = ['Ag','Au', 'Cu', 'Pd','Pt']

# Define relative blocking vectors
block_fcc_vectors = np.array([[-1,0],[0,-1],[0,0]])
block_top_vectors = np.array([[0,0],[0,1],[1,0]])
    
def block_indices(i,j,ads_site):
    if ads_site=="top":
        block_vectors = np.array([i,j]) + block_fcc_vectors
    elif ads_site=="fcc":
        block_vectors = np.array([i,j]) + block_top_vectors
    block_vectors[block_vectors==100] = 0
    return block_vectors
        
        
np.random.seed(1)        

#composition = np.array([0.2,0.2,0.2,0.2,0.2])
#composition = np.array([0.0,0.5,0.5,0.0,0.0])
composition = np.array([0.0,0.0,1.0,0.0,0.0])

P_CO=1
P_NO=0.1
H_UPD=True
eU=0
replacement_factor=1.0
iterations=int(8e+5)
rate=0.999999

#Initiate surface
surface = initiate_surface(composition,metals)

#Predict energies of every site
CO_energies,CO_site_ids=predict_energies(surface, "CO", metals)
NO_energies,NO_site_ids=predict_energies(surface, "NO_fcc", metals)
H_energies,H_site_ids = predict_energies(surface,"H",metals)


#Adjust free energy by chemical potential from preassure
CO_energies += kBT * np.log(1/P_CO)
NO_energies += kBT * np.log(1/P_NO)

#Turn into grids
CO_energy_grid = CO_energies.reshape(100,100)
NO_energy_grid = NO_energies.reshape(100,100)
H_energy_grid = H_energies.reshape(100,100)

CO_data = np.hstack((CO_energies.reshape(-1,1),CO_site_ids))
NO_data = np.hstack((NO_energies.reshape(-1,1),NO_site_ids))
H_data = np.hstack((H_energies.reshape(-1,1),H_site_ids))

#Only keep negative energies
CO_data = CO_data[CO_data[:,0]<0]
NO_data = NO_data[NO_data[:,0]<0]

#Coverage masks
CO_coverage_mask = np.full((100, 100),None)
NO_coverage_mask = np.full((100, 100),None)
H_coverage_mask = np.full((100, 100),None)


if H_UPD:
    #fill surface with H where adsorption energy is negative
    for (energy, i, j) in H_data:
        i,j=int(i),int(j)
        if energy<=(-eU):
            H_coverage_mask,CO_coverage_mask,NO_coverage_mask = adsorb_H(i,j,H_coverage_mask, CO_coverage_mask, NO_coverage_mask)



P = P_CO + P_NO

p_CO = P_CO/P
p_NO=P_NO/P
energies = np.empty(0)
D_energies = np.empty(0)
it_list=[]
n_reactions = 0
CO_NO_pair=np.empty((0,4))
d_reactions = 0
D_reactions_list = []

for iteration in tqdm(range(int(3e+5))):

    r = np.random.rand()
    if r<= p_CO:

        ind = np.random.choice(len(CO_data))#,p=p)
        energy, i, j = CO_data[ind]
        i,j = int(i),int(j)
        #if CO_coverage_mask[i,j] and iteration%1000!=0: continue
        #CO_data = np.delete(CO_data, ind,axis=0)
        block_vectors = block_indices(i, j, "top")
        #NO_blocked = np.any([NO_coverage_mask[ib,jb] for (ib,jb) in block_vectors])
        if H_UPD:
            H_blocked = np.any([H_coverage_mask[ib,jb] for (ib,jb) in block_vectors])
            if H_blocked and CO_coverage_mask[i,j] is None: print("error")
            if H_blocked: continue
        

        if CO_coverage_mask[i,j] is None:
            CO_coverage_mask,NO_coverage_mask=adsorb_CO(i, j, CO_coverage_mask, NO_coverage_mask)

       
        elif CO_coverage_mask[i,j]==False and np.random.rand() <= replacement_factor:
            
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
                    i_ub,j_ub = block_indices(i_b, j_b, "fcc").T
                    CO_coverage_mask[i_ub,j_ub]=None
                CO_coverage_mask,NO_coverage_mask=adsorb_CO(i, j, CO_coverage_mask, NO_coverage_mask)
                #if Delta_E>0: print(Delta_E,np.exp(-Delta_E/kBT))
    else:        
        ind = np.random.choice(len(NO_data))#,p=p)
        energy, i, j = NO_data[ind]
        i,j = int(i),int(j)
        #if NO_coverage_mask[i,j] and iteration%1000!=0: continue
        #NO_data = np.delete(NO_data, ind,axis=0)
        block_vectors = block_indices(i, j, "fcc")
        #CO_blocked = np.any([CO_coverage_mask[ib,jb] for (ib,jb) in block_vectors])
        if H_UPD:
            H_blocked = H_coverage_mask[i,j]==True
            if H_blocked and CO_coverage_mask[i,j] is None: print("error")
            if H_blocked: continue


        if NO_coverage_mask[i,j] is None:
            CO_coverage_mask,NO_coverage_mask=adsorb_NO(i, j, CO_coverage_mask, NO_coverage_mask)
        
        #elif NO_coverage_mask[i,j]==False and CO_blocked:
        elif NO_coverage_mask[i,j]==False and np.random.rand() <= replacement_factor:
                
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
                        i_ub,j_ub = block_indices(i_b, j_b, "top").T
                        NO_coverage_mask[i_ub,j_ub]=None
                    CO_coverage_mask,NO_coverage_mask=adsorb_NO(i, j, CO_coverage_mask, NO_coverage_mask)
    
    
    
    CO_coverage_bool = np.asanyarray(CO_coverage_mask,dtype=bool)
    NO_coverage_bool = np.asanyarray(NO_coverage_mask,dtype=bool)
    H_coverage_bool = np.asanyarray(H_coverage_mask,dtype=bool)
    
    
    #Pad grids
    NO_coverage_mask_pad = np.pad(NO_coverage_bool,pad_width=((1,1),(1,1)),mode="wrap")
    NO_energy_grid_pad = np.pad(NO_energy_grid,pad_width=((1,1),(1,1)),mode="wrap")


    pairs1_bool=CO_coverage_bool * NO_coverage_mask_pad[:-2,:-2]
    pairs2_bool=CO_coverage_bool * NO_coverage_mask_pad[2:,:-2]
    pairs3_bool=CO_coverage_bool * NO_coverage_mask_pad[:-2,2:]

    pairs1_ind=np.array(np.where(pairs1_bool))
    pairs2_ind=np.array(np.where(pairs2_bool))
    pairs3_ind=np.array(np.where(pairs3_bool))

    pairs1_ind_NO = pairs1_ind + np.array([[-1],[-1]])
    pairs2_ind_NO = pairs2_ind + np.array([[1],[-1]])
    pairs3_ind_NO = pairs3_ind + np.array([[-1],[1]])

    CO_NO_pair1 = np.vstack((pairs1_ind,pairs1_ind_NO))
    CO_NO_pair2 = np.vstack((pairs2_ind,pairs2_ind_NO))
    CO_NO_pair3 = np.vstack((pairs3_ind,pairs3_ind_NO))

    CO_NO_pair = np.hstack((CO_NO_pair1,CO_NO_pair2,CO_NO_pair3))
    
    n_pairs=len(CO_NO_pair.T)
    
    CO_NO_pair[CO_NO_pair==100] = 0
    if n_pairs>0:
        
        p_reaction = np.random.random(size=n_pairs)
        reaction_bool = p_reaction > rate

        #remove reacted pairs
        CO_NO_pair_reacted = CO_NO_pair.T[reaction_bool]
        
        
        
        if len(CO_NO_pair_reacted)>0:
            
            #Done let the same CO or NO react twice
            np.random.shuffle(CO_NO_pair_reacted) #Shuffle to make it random
            NO_react_,NO_u_ind = np.unique(CO_NO_pair_reacted[:,:2],axis=0,return_index=True)
            CO_react_,CO_u_ind = np.unique(CO_NO_pair_reacted[:,2:],axis=0,return_index=True)    
            
            if np.all(CO_u_ind==NO_u_ind)==False:
                ind = [i for i in range(len(CO_NO_pair_reacted)) if (i in CO_u_ind and i in NO_u_ind)]
                CO_NO_pair_reacted = CO_NO_pair_reacted[ind]
            
            
            
            i_CO,j_CO,i_NO,j_NO= CO_NO_pair_reacted.T
            
            CO_coverage_mask[i_CO,j_CO] = None
            NO_coverage_mask[i_NO,j_NO] = None
            for (i_CO,j_CO,i_NO,j_NO) in CO_NO_pair_reacted:
                unblock_NO = block_indices(i_CO, j_CO, "top")
                unblock_CO = block_indices(i_NO, j_NO, "fcc")
                ib,jb = unblock_CO.T
                CO_coverage_mask[ib,jb] = None
                ib,jb = unblock_NO.T
                NO_coverage_mask[ib,jb] = None
                
        new_reactions = len(CO_NO_pair_reacted)
    
    else:   
        new_reactions = 0
        
    n_reactions += new_reactions
    d_reactions += new_reactions
    if iteration%10000==0:
        
        D_reactions= d_reactions/10000
        D_reactions_list.append(D_reactions)
        d_reactions=0
        it_list.append(iteration)
        #if abs(D_reactions[-1] - D_reactions[-2])<10: 
         #   break

                
                
            
    """
    if iteration%1000==0:
        CO_ads = CO_energy_grid[np.asanyarray(CO_coverage_mask,dtype=bool)]
        NO_ads = NO_energy_grid[np.asanyarray(NO_coverage_mask,dtype=bool)]
        H_ads = H_energy_grid[np.asanyarray(H_coverage_mask,dtype=bool)]
        energies=np.append(energies,np.sum(CO_ads)+np.sum(NO_ads)+np.sum(H_ads))
        it_list.append(iteration)
        if len(energies)>1:
            D_energies = np.append(D_energies,(energies[-1]-energies[-2])/1000)
            """
            
            
print(n_reactions)




surface_atoms = np.multiply(*surface.shape[:2])
n_CO_NO_pairs = len(CO_NO_pair.T)



CO_ads = CO_energy_grid[CO_coverage_bool]
NO_ads = NO_energy_grid[NO_coverage_bool]
H_ads = H_energy_grid[H_coverage_bool]
plt.figure()
plt.hist(CO_ads,bins=100,histtype="step",label="CO")
plt.hist(NO_ads,bins=100,histtype="step",label="NO")
plt.hist(H_ads,bins=100,histtype="step",label="H")
plt.legend()

print(np.sum(CO_ads)+np.sum(NO_ads)+np.sum(H_ads))


fig,ax=plt.subplots()
#ax.plot(it_list,energies)
ax.plot(it_list,D_reactions_list)
#ax2=ax.twinx()
#ax2.plot(it_list[1:],D_energies,c="tab:orange")
ax.set_xscale("log")

print(D_reactions_list[-10:])

