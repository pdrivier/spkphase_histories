#bootstrapping only script



## Load relevant libraries
import itertools
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import seaborn as sns

from Automaze_Spkphase import Automaze_Spkphase
from collections import Counter
from crossval_fns import train_test_split_bytrials
from mdl_eval_tools import bayes, ksPlot, logloss, logodds, logistic
from process_trial_data import find_true_trial_end_samples
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

def label_phase_cycles(df,df_col_name_phase,df_col_name_cyclelabels):
    """df_col_name, string, e.g. 'phase"""

    df['phase_max2pi'] = df[df_col_name_phase] + np.pi

    st=df['phase_max2pi'].iloc[0]

    true_inds_within_cycle = np.where((df['phase_max2pi'] <= st))
    large_jumps = np.where(np.diff(true_inds_within_cycle[0])>1)

    cycle_edges = true_inds_within_cycle[0][large_jumps]

    if cycle_edges[0] != 0:
        cycle_edges = np.hstack((0,cycle_edges,df.index[-1]))
    else:
        cycle_edges = np.hstack((cycle_edges,df.index[-1]))

    labels = np.linspace(0,len(cycle_edges),len(cycle_edges)-1)

    df[df_col_name_cyclelabels] = pd.cut(df.index,
                                         bins=cycle_edges,
                                         labels=labels,
                                         include_lowest=True,
                                         duplicates='drop'
                                        )
    df[df_col_name_cyclelabels] = df[df_col_name_cyclelabels].astype(int)

    return df

# LOAD DATA:grab from previous runs with real cells, so you can keep the training trials constant
rhythm = 'theta'
longhistlen = 250


N_FOLDS = np.arange(5,6)
N_PERMS = 100

for fold in N_FOLDS:

    data_path = 'FINAL/Entropy2020_Fig4_Dataset/' + 'logoddsCompleteSinglePredMdls1061/' + rhythm + str(longhistlen) + 'ms' + '/folds/' + str(fold) + '/single_neuron/'

    data_folders=[]
    data_folders = [f for f in os.listdir(data_path) if not f.startswith('.')]
    data_folders = sorted(data_folders)

    NEURON_SAVEPATH = 'FINAL/Entropy2020PhasePermDataset/' + 'logoddsCompleteSinglePredMdls1061/' + rhythm + str(longhistlen) + 'ms' + '/folds/' + str(fold) +  '/single_neuron/'



    for n in tqdm(range(len(data_folders))):

        nrn_path = os.path.join(data_path,data_folders[n])

        train_data = pd.read_csv(os.path.join(nrn_path,'train_data.csv'))

        train_data = label_phase_cycles(train_data,'phase','cycle_id')

        cycle_id = set(train_data['cycle_id'].to_list())

        for p in range(0,N_PERMS):
            col_name = 'phaseperm' + str(p)

            shuffled_phases=[]
            for c in cycle_id:
                sub = train_data['phase'][train_data['cycle_id']==c]
                shuffled_phases = np.hstack((shuffled_phases,np.random.permutation(sub.values)))

            train_data[col_name] = shuffled_phases

        # save to csvs
        neuron_savepath = os.path.join(NEURON_SAVEPATH,data_folders[n])
        if not os.path.exists(neuron_savepath):
            os.makedirs(neuron_savepath)
        train_data.to_csv(os.path.join(neuron_savepath,r'permphase.csv'))
