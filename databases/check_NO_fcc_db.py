import numpy as np
from ase.db import connect
from ase.data import covalent_radii,vdw_radii
import matplotlib.pyplot as plt
from collections import Counter

not_ontop=0

z_vec = np.array([0,0,1])
angles = np.empty(0)
not_ontop_angles = np.empty(0)
ontop_angles = np.empty((0,2))

metals = ['Ag','Au', 'Cu', 'Pd','Pt']

sites=[]
fcc_top_dist = []

def get_site_pos_in_xy(pos_1st):
    # Rearange positions so that they are ordered by layer
    pos_1st_ = np.array([pos_1st[(9*3*i)+j*3:3+3*9*i+j*3] for j in range(9) for i in range(3)])
    pos_1st_ = pos_1st_.reshape(81,3)
    #print(pos_1st_)
    grid = pos_1st_.reshape(9,9,-1)
    #print(grid)
    grid = np.pad(grid,pad_width=((1,1),(1,1),(0,0)),mode="wrap")
    
    fcc_sites =  (grid[1:-1,1:-1] + grid[1:-1,2:] + grid[2:,1:-1])/3
    
    hcp_sites = (grid[1:-1,1:-1] + grid[2:,:-2] +grid[2:,1:-1])/3
    
    bridge_sites1 = (grid[1:-1,1:-1] + grid[1:-1,2:])/2
    bridge_sites2 = (grid[1:-1,1:-1] + grid[2:,1:-1])/2
    bridge_sites3 = (grid[1:-1,1:-1] + grid[2:,:-2])/2
    
    
    # cut off ends as we are only interested in sites in the middle 3x3 atoms anyway
    fcc_sites = fcc_sites[2:-2,2:-2]
    hcp_sites = fcc_sites[2:-2,2:-2]
    bridge_sites1 = bridge_sites1[2:-2,2:-2]
    bridge_sites2 = bridge_sites2[2:-2,2:-2]
    bridge_sites3 = bridge_sites3[2:-2,2:-2]
    ontop_sites = np.copy(grid[3:-3,3:-3]).reshape(-1,3)
    
    bridge_sites = np.vstack([bridge_sites1.reshape(-1,3),bridge_sites2.reshape(-1,3),bridge_sites3.reshape(-1,3)])
    return fcc_sites.reshape(-1,3), hcp_sites.reshape(-1,3), bridge_sites,ontop_sites

def get_nearest_sites_in_xy(fcc,hcp,bridge,ontop,ads):
    fcc_dist = np.sum((fcc[:,:2]-ads[:2])**2,axis=1)
    hcp_dist = np.sum((hcp[:,:2]-ads[:2])**2,axis=1)
    bridge_dist = np.sum((bridge[:,:2]-ads[:2])**2,axis=1)
    ontop_dist = np.sum((ontop[:,:2]-ads[:2])**2,axis=1)
    
    min_ids = [np.argmin(dist) for dist in (fcc_dist,hcp_dist,bridge_dist,ontop_dist)]
    min_dists = [dist[min_ids[i]] for i,dist in enumerate((fcc_dist,hcp_dist,bridge_dist,ontop_dist))]
    
    site_str = ["fcc","hcp","bridge","ontop"]
    
    nearest_site_type = site_str[np.argmin(min_dists)]
    
    return nearest_site_type, min_ids
    


with connect("NO_fcc_out.db") as db:
    for row in db.select():
        if row.formula =="":
            sites.append("empty")
            continue
        atoms=db.get_atoms(row.id)
        site_idx=row.site_idx
        
        # Repeat atoms object
        atoms_3x3 = atoms.repeat((3,3,1))
		
		# Get chemical symbols of atoms
        symbols = np.asarray(atoms_3x3.get_chemical_symbols())
		
		# Get index of central hydrogen atom
        idx_N = np.nonzero(symbols == 'N')[0][4]
		
		# Get position of carbon atom
        pos_N = atoms_3x3.positions[idx_N]
		
		# Get indices of the 1st layer atoms
        ids_1st = np.array([atom.index for atom in atoms_3x3 if atom.tag == 1])
		
        # Get posittion of atoms in 1st layer
        pos_1st=atoms_3x3.positions[ids_1st]
        
		# Get squared distances to the 1st layer atoms from the hydrogen atom
        dists_sq = np.sum((pos_1st - pos_N)**2, axis=1)
		
        # Get the three shortest distances
        dist_3 = np.sort(np.sqrt(dists_sq))[:3]  
        
        # Get one atom from 4th and 5th layer 
        id_4th = np.array([atom.index for atom in atoms_3x3 if atom.tag == 4])[0]
        id_5th = np.array([atom.index for atom in atoms_3x3 if atom.tag == 5])[7]
        
        # Get distance in z
        z_dist = atoms_3x3.positions[id_4th][2] - atoms_3x3.positions[id_5th][2]
        
        # Set average radius of atoms
        r = z_dist/2
        # Get radius of N
        r_N = covalent_radii[7]
        
        #Set reference distance
        r_ref = r + r_N
        
        #Get number of distances below reference +20%
        n = np.sum(dist_3 < r_ref*1.2)
        
        
        #Get all site positions
        fcc_sites_pos,hcp_sites_pos,bridge_sites_pos,ontop_sites_pos = get_site_pos_in_xy(pos_1st)
        
        nearest_site,min_dist_ids=get_nearest_sites_in_xy(fcc_sites_pos, hcp_sites_pos, bridge_sites_pos, ontop_sites_pos, pos_N)
        #print(fcc_sites_pos[:,:2])
        site_dists = np.linalg.norm(fcc_sites_pos[:,:2] - ontop_sites_pos[:,:2],axis=1)
        #print(site_dists)
        
        fcc_top_dist.append(np.mean(site_dists))
        
        
        if nearest_site=="bridge":
            fcc_dist = np.linalg.norm(fcc_sites_pos[min_dist_ids[0]][:2] - pos_N[:2])
            hcp_dist =np.linalg.norm(hcp_sites_pos[min_dist_ids[1]][:2] - pos_N[:2])
            top_dist =np.linalg.norm(ontop_sites_pos[min_dist_ids[3]][:2] - pos_N[:2])
            site_names = ["fcc","hcp","ontop"]
            nearest_site = site_names[np.argmin([fcc_dist,hcp_dist,top_dist])]
        sites.append(nearest_site)
        
        """
        # Get index of central Oxygen atom
        idx_O = np.nonzero(symbols == 'O')[0][4]
		
		# Get position of carbon atom
        pos_O = atoms_3x3.positions[idx_O]
        
        NO_vec = pos_O - pos_N
        
        angle = np.arccos(np.dot(z_vec,NO_vec)/(np.linalg.norm(NO_vec)*np.linalg.norm(z_vec)))
        angle*=180/np.pi
        
        
        if n>1:
            not_ontop+=1
            #print(row.id,row.site_idx)
            not_ontop_angles = np.append(not_ontop_angles,angle)
        else:
            
            idx_site = ids_1st[np.argmin(dists_sq)]
            
            metal = symbols[idx_site]
            append = np.array([[angle,metals.index(metal)]])
            ontop_angles = np.vstack((ontop_angles,append))
            """
print(sites[:20])
print(Counter(sites))
plt.hist(fcc_top_dist)
print(np.mean(fcc_top_dist))
stop
#print(ontop,bridge,hollow)
#print(sum([ontop,bridge,hollow]))
#print("not ontop:",not_ontop)
plt.figure(dpi=400)
plt.hist(ontop_angles[:,0],bins=20,histtype="step",color="tab:blue",label="Ontop sites")
plt.hist(not_ontop_angles,bins=20,histtype="step",color="tab:orange",label= "Not ontop sites")
plt.legend()
plt.xlabel("N-O angle (degrees)")
plt.ylabel("Counts")

plt.figure(dpi=400)
#mask_Au = ontop_angles[:,1]==0
#mask_Ag = ontop_angles[:,1]==1
#mask_Cu = ontop_angles[:,1]==2
#mask_Pd = ontop_angles[:,1]==3
#mask_Pt = ontop_angles[:,1]==4

for i,metal in enumerate(metals):
    mask = ontop_angles[:,1]==i
    angles = ontop_angles[:,0][mask]
    plt.hist(angles,bins=20,histtype="step",label=metal)
    print(metal,": Mean angle: ", np.mean(angles))
plt.legend()
plt.xlabel("N-O angle (degrees)")
plt.ylabel("Counts")

print("\nsingle element slabs")
#single element slabs
with connect("single_element_NO_angled_out.db") as db:
    for row in db.select():
        metal = row.metal
        atoms=db.get_atoms(row.id)
        
        # Repeat atoms object
        atoms_3x3 = atoms.repeat((3,3,1))
		
		# Get chemical symbols of atoms
        symbols = np.asarray(atoms_3x3.get_chemical_symbols())
		
		# Get index of central hydrogen atom
        idx_N = np.nonzero(symbols == 'N')[0][4]
		
		# Get position of carbon atom
        pos_N = atoms_3x3.positions[idx_N]
		
		# Get indices of the 1st layer atoms
        ids_1st = np.array([atom.index for atom in atoms_3x3 if atom.tag == 1])
		
		# Get squared distances to the 1st layer atoms from the hydrogen atom
        dists_sq = np.sum((atoms_3x3.positions[ids_1st] - pos_N)**2, axis=1)
		
        # Get the three shortest distances
        dist_3 = np.sort(np.sqrt(dists_sq))[:3]  
        
        # Get index of central Oxygen atom
        idx_O = np.nonzero(symbols == 'O')[0][4]
		
		# Get position of carbon atom
        pos_O = atoms_3x3.positions[idx_O]
        
        NO_vec = pos_O - pos_N
        print(np.linalg.norm(NO_vec))
        angle = np.arccos(np.dot(z_vec,NO_vec)/(np.linalg.norm(NO_vec)*np.linalg.norm(z_vec)))
        angle*=180/np.pi
        
        print(metal,": angle: ",angle)
