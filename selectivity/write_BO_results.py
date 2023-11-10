import numpy as np

for P_NO in ['1','0.1']:

    P_NO_str = P_NO.replace('.','')
    # Load data
    data = np.loadtxt(f'Bayesian_optimization_selectivity_results_PNO_{P_NO_str}.csv',delimiter=',',skiprows=1)

    # Sort data
    data_sorted = data[np.argsort(-data[:,-1])]

    # Composition with best selectivity
    mfs = data_sorted[0,:5].reshape(1,-1)
    n_sites = data_sorted[0,-1]
    
    # Append data that is min 10 at% away from max
    for dataline in data_sorted[1:]:
        mf = dataline[:-1]
        ns = dataline[-1]
        dist = np.linalg.norm(mfs-mf,axis=1)

        if np.all(dist>0.1):
            mfs = np.vstack((mfs,mf))
            n_sites = np.append(n_sites,ns)

    
    print(f'P_NO={P_NO}:')

    for mf, ns in zip(mfs[:3],n_sites[:3]):
        print(mf,ns)

    