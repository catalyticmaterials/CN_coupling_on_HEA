import numpy as np
import matplotlib.pyplot as plt

#Load energies
path = "Ag0.2Au0.2Cu0.2Pd0.2Pt0.2/"
CO=np.loadtxt(path+"CO_energies.csv",delimiter=" ")
NO=np.loadtxt(path+"NO_energies.csv",delimiter=" ")

#Append index to data
CO_id = np.ones((len(CO),1))
NO_id = np.ones((len(NO),1))*2
CO = np.hstack((CO,CO_id))
NO = np.hstack((NO,NO_id))

plt.hist(CO[:,0],histtype="step",bins=100)
plt.hist(NO[:,0],histtype="step",bins=100)




#Combine data
data_array=np.vstack((CO,NO))

#Sort data by lowest energy
data_array = data_array[data_array[:, 0].argsort()]

#Prepare grids
fcc_grid = np.zeros((100,100))
top_grid = fcc_grid.copy()

CO_energies = np.empty(0)
NO_energies = np.empty(0)


block_fcc_vectors = np.array([[-1,0],[0,-1],[0,0]])
block_top_vectors = np.array([[-1,0],[-1,1],[1,0]])

for (energy,i,j,idx) in data_array:
    if energy>0: break
    i,j,idx = int(i),int(j),int(idx)
    ij_vec = np.array([i,j])
    
    

        
    if idx==1:
        if top_grid[i,j]==0:
            top_grid[i,j] = energy
            #CO_energies = np.append(CO_energies,energy)
            
            #Block sites
            ij_block_vectors = ij_vec + block_fcc_vectors
            for (i_b,j_b) in ij_block_vectors:
                fcc_grid[i_b,j_b] = 1
        else:
            continue
    
    elif idx==2:
        if fcc_grid[i,j]==0:
            fcc_grid[i,j] = energy
            #NO_energies = np.append(NO_energies,energy)
            
            #Block sites
            ij_block_vectors = ij_vec + block_top_vectors
            for (i_b,j_b) in ij_block_vectors:
                if i_b == 100:
                    i_b=0
                elif j_b == 100:
                    j_b=0
                top_grid[i_b,j_b] = 1
        
        
        



CO_ids = np.array(np.nonzero(top_grid<0)).T
ads_energy_pair = np.empty((0,2))

#pad grids
fcc_grid = np.pad(fcc_grid,pad_width=1,mode="wrap")
top_grid = np.pad(top_grid,pad_width=1,mode="wrap")
CO_ids+=1


#Get all pairs of catalytic sites
for (i,j) in CO_ids:
    if fcc_grid[i-1,j-1] < 0:
        E_CO = E_CO = top_grid[i,j]#CO_energies[i*100+j]
        E_NO = fcc_grid[i-1,j-1] #NO_energies[(i-1)*100+(j-1)]
        ads_energy_pair = np.vstack((ads_energy_pair,np.array([[E_CO,E_NO]])))
    if fcc_grid[i-1,j+1] < 0:
        E_CO = E_CO = top_grid[i,j]#CO_energies[i*100+j]
        E_NO = fcc_grid[i-1,j+1] #NO_energies[(i-1)*100+(j+1)]
        ads_energy_pair = np.vstack((ads_energy_pair,np.array([[E_CO,E_NO]])))
    if fcc_grid[i+1,j-1] < 0:
        E_CO = top_grid[i,j] #CO_energies[i*100+j]
        E_NO = fcc_grid[i+1,j-1] #NO_energies[(i+1)*100+(j-1)]
        ads_energy_pair = np.vstack((ads_energy_pair,np.array([[E_CO,E_NO]])))

print(len(ads_energy_pair))
print(len(CO_ids))
"""
plt.figure(dpi=400)
plt.hist(CO_energies,histtype="step",label="CO",bins=100)
plt.hist(NO_energies,histtype="step",label="NO",bins=100)
plt.legend(loc=2)
plt.text(-1.4,300,f"Nr. of CO sites: {len(CO_energies)}\nNr. of NO sites: {len(NO_energies)}")




plt.figure(dpi=400)
plt.scatter(CO[:,0],NO[:,0],marker=".")
plt.plot(CO[:,0],CO[:,0],c="k")
"""
