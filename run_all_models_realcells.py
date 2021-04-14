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
# from kde_history_models_20210312 import kdePhase_logregHistory_models
from kde_history_models_20210324_l2penaltyforlonghistories import kdePhase_logregHistory_models
from mdl_eval_tools import bayes, ksPlot, logloss, logodds, logistic
from process_trial_data import find_true_trial_end_samples
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

# LOAD DATA
data_path = "python_spkphase_multirhythm/"

data_folders=[]
data_folders = [f for f in os.listdir(data_path) if not f.startswith('.')]
data_folders = sorted(data_folders)

rhythm = 'theta'
longhistlen = 250

# SET BANDWITH RANGE FOR KERNEL DENSITY ESTIMATION ## Select the bandwidth range
# for kernelÔ
NARROW = 1/30
WIDE = 1/0.6
N_BW = 20
kde_cvsplits = 5
kde_gridsize = 1000 #for WARPing estimation method

# DEFINE DATASET PARAMS
fs = 1000 #sampling rate, in Hz

# DEFINE variable names to grab from .mat file
spikes_var_name = 'spikes_long'
phase_var_name = rhythm +'phase_long'
lfp_var_name = 'lfp_long'

#cross validation test set size
test_size = 0.5

# decide how many folds of the data you will run
N_FOLDS = [16,17,18, 19, 20]

#decide whether you'll do complete models with logodds of both history and phase probs,
#rather than raw history data
rng = np.random.default_rng()

logodds_completemdls = 1

if logodds_completemdls == 1:
    subfolder_name = 'logoddsCompleteSinglePredMdls'+ str(rng.integers(low=0,high=1300)) + '/'
else:
    subfolder_name = 'logoddsCompleteMdls' + str(rng.integers(low=0,high=1300)) + '/'


for fold in N_FOLDS:
    MAIN_SAVEPATH = 'FINAL/Entropy2020PhaseHistModels_AllCells/' + subfolder_name + rhythm + str(longhistlen) + 'ms' + '/folds/' + str(fold)
    NEURON_SAVEPATH =  MAIN_SAVEPATH + '/single_neuron/'


    # PREALLOCATE VARS TO STORE POPULATION DATA
    coeff_clong = []
    coeff_TRANS_hlong = []
    coeff_TRANS_hshort = []
    coeff_cshort = []

    kde_bandwidths = []
    pop_losses_train = []
    pop_losses_test = []

    for n in tqdm(range(len(data_folders))):#   range(len(data_folders)) np.arange(0,1))


        nrn_path = os.path.join(data_path,data_folders[n])

        for file in os.listdir(nrn_path):
            if not file.startswith('.') & file.endswith('.mat'):
                file_name = file.split('.mat')[0]
                file_path = os.path.join(nrn_path, file_name)

        session_name = file_path + '_results'

        data = sio.loadmat(file_path,squeeze_me=1)

        session = Automaze_Spkphase(data,
                          session_id=session_name,
                          pre_trial_time=.250,
                          fs=fs,
                          long=longhistlen/1000,
                          short=0.003)

        hist_longest = session.extra_msec()

        All_Spikes, All_Phases, All_LFP = session.get_fulldata(spikes_var_name,phase_var_name,lfp_var_name)

        trial_data, iter_timesamps, trial_ends, n_trials = session.process_mat_struct()

        n_trials = trial_data.get('n_trials')

        histlong, histshort = session.history_in_msec()

        df, Trial_Spikes, Trial_Phases, Trial_LFP= session.make_history()

        df = session.label_trials()

        hist_long_col = list(reversed(range(hist_longest-histlong,hist_longest)))
        hist_short_col = list(reversed(range(hist_longest-histshort,
                                             hist_longest)))

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

        nrn_savepath = os.path.join(NEURON_SAVEPATH,data_folders[n])
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
