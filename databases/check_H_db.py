import numpy as np
from ase.db import connect
from ase.data import covalent_radii,vdw_radii

fcc=0
hcp=0
not_hollow = 0


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
    bridge_sites = np.vstack([bridge_sites1.reshape(-1,3),bridge_sites2.reshape(-1,3),bridge_sites3.reshape(-1,3)])
    
    ontop_sites = np.copy(pos_1st)
    
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
    
    

with connect("H_out.db") as db:
    for row in db.select():
        if row.formula=="":
            continue
        atoms=db.get_atoms(row.id)
        site_idx=row.site_idx
        
        # Repeat atoms object
        atoms_3x3 = atoms.repeat((3,3,1))
		
		# Get chemical symbols of atoms
        symbols = np.asarray(atoms_3x3.get_chemical_symbols())
		
		# Get index of central hydrogen atom
        idx_H = np.nonzero(symbols == 'H')[0][4]
		
		# Get position of carbon atom
        pos_H = atoms_3x3.positions[idx_H]
		
		# Get indices of the 1st layer atoms
        ids_1st = np.array([atom.index for atom in atoms_3x3 if atom.tag == 1])

        # Get posittion of atoms in 1st layer
        pos_1st=atoms_3x3.positions[ids_1st]
        
		# Get squared distances to the 1st layer atoms from the hydrogen atom
        dists_sq = np.sum((pos_1st - pos_H)**2, axis=1)
		
		# Get the closest 3 surface atoms to the hydrogen
		# (the three-fold hollow adsorption site)
        ids_site = ids_1st[np.argpartition(dists_sq, kth=2)[:3]]
        #print(atoms_3x3.positions[ids_site])
		# Get the position of the adsorption site as the mean of the positions
		# of the atoms in the site
        pos_site = np.mean(atoms_3x3.positions[ids_site], axis=0)

		# Get indices of the 2nd layer atoms
        ids_2nd = np.array([atom.index for atom in atoms_3x3 if atom.tag == 2])
		
		# Get squared distances from adsorption site to the atoms in each layer
        dists_sq_1st = np.sum((atoms_3x3.positions[ids_1st] - pos_site)**2, axis=1)
        dists_sq_2nd = np.sum((atoms_3x3.positions[ids_2nd] - pos_site)**2, axis=1)
		
        # Get the three shortest distances
        #dist_3 = np.sort(np.sqrt(dists_sq))[:3]  
        
        # Get one atom from 4th and 5th layer 
        id_4th = np.array([atom.index for atom in atoms_3x3 if atom.tag == 4])[0]
        id_5th = np.array([atom.index for atom in atoms_3x3 if atom.tag == 5])[0]
        
        # Get distance in z
        z_dist = atoms_3x3.positions[id_4th][2] - atoms_3x3.positions[id_5th][2]
        
        min_dist_2nd = np.sqrt(np.min(dists_sq_2nd))
		
        #print(np.sort(np.sqrt(dists_sq_2nd))[:3],z_dist*1.1)
        #break
        
        site_dist_xy = np.sqrt(np.sum((pos_H[:2]-pos_site[:2])**2))
        
        #Check for hcp vs fcc
        if min_dist_2nd < z_dist*1.1:
            #If far from hcp site making on-top: treat as nearest fcc hollow site
            if site_dist_xy>(z_dist/2):
                not_hollow+=1
                print("not hollow:",row.id)
            else: 
                hcp+=1
                print("hcp",row.id)

        else:
            fcc+=1

            fcc_sites_pos,hcp_sites_pos,bridge_sites_pos,ontop_sites_pos = get_site_pos_in_xy(pos_1st)
            
            nearest_site,min_dist_ids=get_nearest_sites_in_xy(fcc_sites_pos, hcp_sites_pos, bridge_sites_pos, ontop_sites_pos, pos_H)
            

            nearest_fcc_site=fcc_sites_pos[min_dist_ids[0]]
            print(nearest_fcc_site)
            break

        
        """
        site_dist_xy = np.sqrt(np.sum((pos_H[:2]-pos_site[:2])**2))
        if site_dist_xy > (z_dist/2):
            not_hollow+=1
            print("not_hollow",row.id)
        elif min_dist_2nd < z_dist*1.1:
            hcp+=1
            print("hcp",row.id)
        else:
            fcc+=1
            """
"""
        # Set average radius of atoms
        r = z_dist/2
        # Get radius of N
        r_N = covalent_radii[7]
        
        #Set reference distance
        r_ref = r + r_N
        
        #Get number of distances below reference +20%
        n = np.sum(dist_3 < r_ref*1.2)

        if n>1:
            not_fcc+=1
            print(row.id)
"""
#print(ontop,bridge,hollow)
#print(sum([ontop,bridge,hollow]))
print(fcc,hcp)
