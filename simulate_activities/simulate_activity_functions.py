import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#from scipy.optimize import curve_fit
from scipy.stats import linregress
#from methods import adsorb_CO,adsorb_H,adsorb_NO
import sys
sys.path.append("..")
from shared_params.surface import predict_energies, initiate_surface
#from shared_params.kinetic_model import fractional_urea_rate, urea_rate,urea_conversion
from time import time

n=50

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
    
def block_indices(i,j,ads_site,n):
    if ads_site=="top":
        block_vectors = np.array([i,j]) + block_fcc_vectors
    elif ads_site=="fcc":
        block_vectors = np.array([i,j]) + block_top_vectors
    block_vectors[block_vectors==n] = 0
    return block_vectors


def adsorb_H(i,j,H_coverage_mask,CO_coverage_mask,NO_coverage_mask,n):
    H_coverage_mask[i,j] = True
    
    i_b,j_b = block_indices(i,j,"fcc",n).T
    CO_coverage_mask[i_b,j_b] = False
    NO_coverage_mask[i,j] = False
    return H_coverage_mask,CO_coverage_mask,NO_coverage_mask

def adsorb_CO(i,j,CO_coverage_mask,NO_coverage_mask,n):
    CO_coverage_mask[i,j] = True
    #CO_energies = np.append(CO_energies,energy)
    
    #Block sites
    i_b,j_b = block_indices(i,j,"top",n).T
    NO_coverage_mask[i_b,j_b] = False
        #H_coverage_mask[i_b,j_b] = False
    return CO_coverage_mask,NO_coverage_mask

def adsorb_NO(i,j,CO_coverage_mask,NO_coverage_mask,n):
    NO_coverage_mask[i,j] = True
    #NO_energies = np.append(NO_energies,energy)
    
    #Block sites
    i_b,j_b = block_indices(i,j,"fcc",n).T
    CO_coverage_mask[i_b,j_b] = False
    #H_coverage_mask[i,j] = False
    return CO_coverage_mask,NO_coverage_mask

def line(x,a,b):
    return a*x+b
        
def simulate_activity(composition,n=50,P_CO=1.0,P_NO=1.0,max_iterations=int(1e+5),rate=0.0,eU=0,H_UPD=True):
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
    #H_energy_grid = H_energies.reshape(n,n)
    
    CO_data = np.hstack((CO_energies.reshape(-1,1),CO_site_ids))
    NO_data = np.hstack((NO_energies.reshape(-1,1),NO_site_ids))
    H_data = np.hstack((H_energies.reshape(-1,1),H_site_ids))
    
    #Only keep negative energies
    CO_data = CO_data[CO_data[:,0]<0]
    NO_data = NO_data[NO_data[:,0]<0]
    
    if len(CO_data)==0 or len(NO_data)==0:
        print(f"{composition}, Rate: 0.0, Eval. time: {time()-t_start}")
        return 0
    
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

    d_reactions = 0
    D_reactions_list = []
    it_list = []
    
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
        
        
        
        CO_coverage_bool = np.asanyarray(CO_coverage_mask,dtype=bool)
        NO_coverage_bool = np.asanyarray(NO_coverage_mask,dtype=bool)
        
        
        
        #Pad grids
        NO_coverage_mask_pad = np.pad(NO_coverage_bool,pad_width=((1,1),(1,1)),mode="wrap")



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
        
        CO_NO_pair[CO_NO_pair==n] = 0
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
                
                #if np.all(CO_u_ind==NO_u_ind)==False:
                ind = [i for i in range(len(CO_NO_pair_reacted)) if (i in CO_u_ind and i in NO_u_ind)]
                CO_NO_pair_reacted = CO_NO_pair_reacted[ind]
                
                
                
                i_CO,j_CO,i_NO,j_NO= CO_NO_pair_reacted.T
                
                CO_coverage_mask[i_CO,j_CO] = None
                NO_coverage_mask[i_NO,j_NO] = None
                for (i_CO,j_CO,i_NO,j_NO) in CO_NO_pair_reacted:
                    unblock_NO = block_indices(i_CO, j_CO, "top",n)
                    unblock_CO = block_indices(i_NO, j_NO, "fcc",n)
                    not_H_blocked = np.array([np.any(H_coverage_bool[block_indices(ib_, jb_, "top",n).T])==False for ib_,jb_ in unblock_CO])
                    ib,jb = unblock_CO[not_H_blocked].T
                    CO_coverage_mask[ib,jb] = None
                    ib,jb = unblock_NO.T
                    not_H_blocked = H_coverage_bool[ib,jb]==False
                    ib,jb = unblock_NO[not_H_blocked].T
                    NO_coverage_mask[ib,jb] = None
                    
            new_reactions = len(CO_NO_pair_reacted)
        
        else:   
            new_reactions = 0
            

        d_reactions += new_reactions
        if (iteration+1)%2000==0:
            
            D_reactions= d_reactions/2000
            D_reactions_list.append(D_reactions)
            d_reactions=0
            it_list.append(iteration)
            
            if iteration>20000:
                mean_D_reactions = np.mean(D_reactions_list[-10:])
                a,*ignore = linregress(it_list[-10:], D_reactions_list[-10:])
                #(a,b),pcov=curve_fit(line, it_list[-10:], D_reactions_list[-10:],sigma=np.sqrt(np.array(D_reactions_list[-10:])))

                if abs(a)<(mean_D_reactions*1e-6):
                    
                    break
                
        
    print(f"{composition}, Rate: {mean_D_reactions:.4f}, Eval. time: {time()-t_start}")
    return mean_D_reactions
                
            