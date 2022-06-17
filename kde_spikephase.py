# Kernel density estimation to obtain spike-phase relationships
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

def kde_spikephase_estimator(df,n_splits,grid_size,bw_narrow,bw_wide,n_bw):

    x = []
    for s,p in zip(df['spikes'].values,df['phase'].values):
        if s==1:
            x.append(p)

    x = np.array(x)
    phase_grid = np.linspace(-np.pi, np.pi, grid_size)

    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': np.linspace(bw_narrow,bw_wide,n_bw)},
                        cv = n_splits)

    #hacky way to get the kernel to treat phase circularly
    cycle1 = x - 2*np.pi
    cycle2 = x
    cycle3 = x + 2*np.pi

    fitting_sample = np.hstack((cycle1, cycle2, cycle3))
    grid.fit(fitting_sample[:,None])

    # print(grid.best_params_)

    kde = grid.best_estimator_
    pdf = np.exp(kde.score_samples(phase_grid[:,None]))
    pdf = pdf * 3 #adjust for the extra data added on either side

    bw = kde.bandwidth
    return pdf, kde, bw, x

def kde_spikephase_estimator_forarray(phi_when_sp,n_splits,grid_size,bw_narrow,bw_wide,n_bw):


    phase_grid = np.linspace(-np.pi, np.pi, grid_size)

    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': np.linspace(bw_narrow,bw_wide,n_bw)},
                        cv = n_splits)

    #hacky way to get the kernel to treat phase circularly
    cycle1 = phi_when_sp - 2*np.pi
    cycle2 = phi_when_sp
    cycle3 = phi_when_sp + 2*np.pi

    fitting_sample = np.hstack((cycle1, cycle2, cycle3))
    grid.fit(fitting_sample[:,None])

    # print(grid.best_params_)

    kde = grid.best_estimator_
    pdf = np.exp(kde.score_samples(phase_grid[:,None]))
    pdf = pdf * 3 #adjust for the extra data added on either side

    bw = kde.bandwidth
    return pdf, kde, bw

def plot_kde(pdf,spkphase_hist, bw, phase_grid,ax=None):
    "Create plot of kde estimator over histogram of spikes by phase"

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))

    ax.plot(phase_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % bw)
    ax.hist(spkphase_hist, 30, fc='gray', histtype='stepfilled', alpha=0.3,density=True)
    ax.legend(loc='upper left')
    # ax.set_xlim(-np.pi, np.pi);


    ax.plot(phase_grid,vm,'g')
    return
