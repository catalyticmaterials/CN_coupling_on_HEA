import numpy as np

def get_site_pos_in_xy(pos_1st):
    # Rearange positions so that they are ordered by layer
    pos_1st_ = np.array([pos_1st[(9*3*i)+j*3:3+3*9*i+j*3] for j in range(9) for i in range(3)])
    pos_1st_ = pos_1st_.reshape(81,3)
    grid = pos_1st_.reshape(9,9,-1)
    grid = np.pad(grid,pad_width=((1,1),(1,1),(0,0)),mode="wrap")
    
    fcc_sites =  (grid[1:-1,1:-1] + grid[1:-1,2:] + grid[2:,1:-1])/3
    
    hcp_sites = (grid[1:-1,1:-1] + grid[2:,:-2] +grid[2:,1:-1])/3
    
    bridge_sites1 = (grid[1:-1,1:-1] + grid[1:-1,2:])/2
    bridge_sites2 = (grid[1:-1,1:-1] + grid[2:,1:-1])/2
    bridge_sites3 = (grid[1:-1,1:-1] + grid[2:,:-2])/2
    
    
    # cut off ends as we are only interested in sites in the middle 3x3 atoms anyway
    fcc_sites = fcc_sites[2:-2,2:-2]
    hcp_sites = hcp_sites[2:-2,2:-2]
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
