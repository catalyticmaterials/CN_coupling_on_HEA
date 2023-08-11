import numpy as np

for method in ['equilibrium']:
    print(method)
    for P_NO in ['1','0.5','0.1']:

        P_NO_str = P_NO.replace('.','')
        data = np.loadtxt(f'{method}/Bayesian_optimization_results_PNO_{P_NO_str}.csv',delimiter=',',skiprows=1)


        

        data_sorted = data[np.argsort(-data[:,-1])]

        mfs = data_sorted[0,:5].reshape(1,-1)
        n_sites = data_sorted[0,-1]
        
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

        