import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import MultipleLocator

N_samples = 10
n_list = [10,50,100,200]

model_dict = {
    'eq': 'Equilibrium',
    'dyn': 'Dynamic',
    'mc': 'Monte Carlo'
}

data = np.loadtxt('model_precisions.csv',delimiter=',',skiprows=1).T

fig, axes = plt.subplots(ncols=2,nrows=3,figsize=(6,6),sharex=True,sharey=True)

i=0
for method,axes_row in zip(['eq','dyn','mc'],axes):
    for P_NO,ax in zip([1,0.1],axes_row):

        means = data[i,:4]
        stds = data[i,4:]

        means_err = stds/np.sqrt(N_samples)

        ax.errorbar(n_list,means,means_err,fmt=".",c="k",label="Mean")

        polygon=Polygon(np.array([np.append(n_list,np.flip(n_list)),np.append(means+stds,np.flip(means-stds))]).T,alpha=0.5,label="std")
        ax.add_patch(polygon)

        #ax.legend(loc=1)
        
        ax.set_xticks(n_list)

        ax.set_ylim(0.0,0.11)

        ax.yaxis.set_major_locator(MultipleLocator(0.02))
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))
 

        ax.text(0.97,0.97,f'{model_dict[method]}\n$P_{{NO}}$={P_NO}',va='top',ha='right',transform=ax.transAxes)


        i+=1

        
handles, labels = axes[0,0].get_legend_handles_labels()

pos1 = axes[0,0].get_position()
pos2 = axes[0,1].get_position()

axes_middle = (pos1.x0 + pos2.x1)/2
print(axes_middle,pos1,pos2)
fig.legend(handles=[handles[1],handles[0]], labels=[labels[1],labels[0]],
           loc='outside upper center', ncol=5, mode='expand',bbox_to_anchor=(axes_middle-0.15, .5, 0.3, 0.5),fancybox=False)

axes[1,0].set_ylabel("Active sites per surface atoms")
axes[2,0].set_xlabel(r"$\sqrt{N_{surface \, atoms}}$")
axes[2,1].set_xlabel(r"$\sqrt{N_{surface \, atoms}}$")



# plt.tight_layout()
fig.subplots_adjust(top=0.93,hspace=0.1,wspace=0.1)
plt.savefig('model_precisions.png',dpi=600,bbox_inches='tight')