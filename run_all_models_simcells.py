## Load relevant libraries
import itertools
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import seaborn as sns

from Automaze_Spkphase import Automaze_Spkphase
from crossval_fns import train_test_split_bytrials
from kde_history_models_20210324_l2penaltyforlonghistories import kdePhase_logregHistory_models
from mdl_eval_tools import bayes, ksPlot, logloss, logodds, logistic
from process_trial_data import find_true_trial_end_samples
from sim_data_tools import make_phase_timeseries
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from tqdm import tqdm


## Select the bandwidth range for kernel
NARROW = 1/30
WIDE = 1/0.6
N_BW = 20
kde_cvsplits = 5
kde_gridsize = 1000 #for WARPing estimation method

fs =1000

sim_filt_freq = 8 #Hz, sets the cycle frequency for the simulated phases

test_size = 0.5

#decide whether you'll do complete models with logodds of both history and phase probs,
#rather than raw history data
rng = np.random.default_rng()

logodds_completemdls = 1

if logodds_completemdls == 1:
    modeltype = 'logoddsCompleteSinglePredMdls'+ str(rng.integers(low=0,high=1300)) + '/'
else:
    modeltype = 'logoddsCompleteMdls' + str(rng.integers(low=0,high=1300)) + '/'


## Pull up the relevant data folders
data_path = 'ScAdv_sim_neuron_data/'
data_folders = []
data_folders = os.listdir(data_path)

for folder in tqdm(data_folders):
    folderpath = os.path.join(data_path,folder,'target_frate/6/')

    data_files = [f for f in os.listdir(folderpath) if not f.startswith('.')]
    data_files = [f for f in data_files if f.endswith('.csv')]
    data_files = sorted(data_files)


    #savepath
    NEURON_SAVEPATH = 'ScAdv_sim_model_fits/' + modeltype + '/' + folder + '/single_neuron/'


    for n in range(len(data_files)):#  range(len(data_folders)))

        nrn_path = os.path.join(folderpath, data_files[n])

        df_og = pd.read_csv(nrn_path)

        #grab number of trials and their lengths
        n_trials = int(df_og['trial_labels'].max()+1)
        len_trials = int(df_og.shape[0]/n_trials)
        true_len_trials = int(len_trials-250)

        #grab the full dataset
        All_Spikes = df_og['spikes'].values

        #simulate a filtered phase dataset
        df_og['sim_filt_phase'] = make_phase_timeseries(sim_filt_freq,df_og.shape[0],fs)
        All_Phases = df_og['sim_filt_phase'].values

        hist_longestsec = .250 #this is the extra time history available during each trial
        hist_longest = int(hist_longestsec*fs)

        n_all_samples = len(All_Spikes) # length of dataset
        n_cut_samples = n_all_samples - hist_longest*n_trials
        #account for lack of continuity between trials
        trial_ends = [(len_trials*i)-1 for i in np.arange(1,n_trials+1)]
        iter_timesamps_lists = [np.arange(i-(len_trials-1),(i-hist_longest)+1) for i in trial_ends]
        iter_timesamps = list(itertools.chain.from_iterable(iter_timesamps_lists))

        #BUILD MASTER HISTORY DataFrame
        Trial_History = np.zeros((hist_longest,n_cut_samples))
        history=[]
        for ind,j in enumerate(iter_timesamps):
            j = int(j)
            history = All_Spikes[j:j+hist_longest]
            Trial_History[:,ind] = history

        histdf = pd.DataFrame(data=Trial_History.T)
        histdf = histdf[histdf.columns[::-1]] #reverse column order

        Trial_Spikes=pd.Series([],name='spikes')
        Trial_Phases=pd.Series([],name='phase')
        for i in trial_ends:
            i = int(i)
            Trial_Spikes = Trial_Spikes.append(pd.Series(data=All_Spikes[i-(true_len_trials-1):i+1]),ignore_index=True)
            Trial_Phases = Trial_Phases.append(pd.Series(data=All_Phases[i-(true_len_trials-1):i+1]),ignore_index=True)

        Trial_Spikes = Trial_Spikes.rename('spikes')
        Trial_Phases = Trial_Phases.rename('phase')

        # create column of trial labels, which will make trial-based cross validation
        # simpler
        true_trial_bins = find_true_trial_end_samples(n_trials,true_len_trials)
        true_trial_bins.insert(0,0)
        labels = np.arange(len(true_trial_bins)-1)

        #BUILD FULL WORKING DATAFRAME WITH ALL AVAILABLE HISTORY
        df = pd.concat([Trial_Phases, Trial_Spikes, histdf],axis=1)

        df['trial_labels'] = pd.cut(df.index.values,bins=true_trial_bins,labels=labels,include_lowest=True)
        ## DEFINE variables for history models

        hist_longsec = 0.250 #for periods in bandpass 4Hz,8Hz,12Hz
        hist_longlag = int(hist_longsec*fs)
        hist_long_col = list(reversed(range(hist_longest-hist_longlag,hist_longest)))


        hist_shortsec = 0.003 #Hodgkin Huxley
        hist_shortlag = int(hist_shortsec*fs)
        hist_short_col = list(reversed(range(hist_longest-hist_shortlag,hist_longest)))

        df, train, test, probs_df_test, probs_df_train, coeff_clong, coeff_hlong, coeff_hshort, coeff_cshort, coeff_TRANS_hshort, coeff_TRANS_hlong, kde_params, kde_bandwidth = kdePhase_logregHistory_models(df,data_folders[n],
                                                                                                                                                                                                            test_size,
                                                                                                                                                                                                            kde_cvsplits,
                                                                                                                                                                                                            kde_gridsize,
                                                                                                                                                                                                            NARROW,
                                                                                                                                                                                                            WIDE,
                                                                                                                                                                                                            N_BW,
                                                                                                                                                                                                            hist_long_col,
                                                                                                                                                                                                            hist_short_col,
                                                                                                                                                                                                            logodds_completemdls)


        #===============================================================================
        #===========create the save folder if it doesn't yet exist======================
        #===============================================================================

        if not os.path.exists(NEURON_SAVEPATH):
            os.makedirs(NEURON_SAVEPATH)

        nrn_savepath = os.path.join(NEURON_SAVEPATH,data_files[n])
        if not os.path.exists(nrn_savepath):
            os.makedirs(nrn_savepath)


        df.to_csv(os.path.join(nrn_savepath,r'raw_data.csv'))
        train.to_csv(os.path.join(nrn_savepath,r'train_data.csv'))
        test.to_csv(os.path.join(nrn_savepath,r'test_data.csv'))
        probs_df_test.to_csv(os.path.join(nrn_savepath,r'probs_models_test.csv'))
        probs_df_train.to_csv(os.path.join(nrn_savepath,r'probs_models_train.csv'))
        coeff_clong.to_csv(os.path.join(nrn_savepath,r'coeff_clong.csv'))
        coeff_cshort.to_csv(os.path.join(nrn_savepath,r'coeff_cshort.csv'))
        coeff_hlong.to_csv(os.path.join(nrn_savepath,r'coeff_hlong.csv'))
        coeff_hshort.to_csv(os.path.join(nrn_savepath,r'coeff_hshort.csv'))

        if coeff_TRANS_hshort.empty & coeff_TRANS_hlong.empty:
            coeff_TRANS_hshort = []
        else:
            coeff_TRANS_hshort.to_csv(os.path.join(nrn_savepath,r'coeff_Transhshort.csv'))
            coeff_TRANS_hlong.to_csv(os.path.join(nrn_savepath,r'coeff_Transhlong.csv'))
