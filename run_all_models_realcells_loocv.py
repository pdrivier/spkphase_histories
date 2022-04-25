## Load relevant libraries
import itertools
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.io as sio
import seaborn as sns

from Automaze_Spkphase import Automaze_Spkphase
from crossval_fns import train_trial_combinations,train_trial_combinations_shuffle
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
# data_path = "python_spkphase_multirhythm/"
data_path = 'python_approach_highestv_data'
# data_path = "python_spkphase_pyrcells_multirhythm_byodorpos"

data_folders=[]
data_folders = [f for f in os.listdir(data_path) if not f.startswith('.')]
data_folders = sorted(data_folders)

#the file structures vary depending on whether you're grabbing interneurons or pyramidal cells
cell_type = 'int'
# cell_type = 'pyr'

# rhythms = ['theta_4_12','beta_15_35','lowgamma_35_55','highgamma_65_90']
rhythms = ['theta_4_12','lowgamma_35_55']

# frequency_edges = [4,12]
longhistlen = 250

# SET BANDWITH RANGE FOR KERNEL DENSITY ESTIMATION ## Select the bandwidth range
# for kernelÔ
NARROW = 1/30
WIDE = 1/0.6
N_BW = 20
kde_gridsize = 1000 #for WARPing estimation method
loocv = 0 #sets whether you will run leave one out cv to find the kde bw (1) or not (0)

# DEFINE DATASET PARAMS
fs = 1000 #sampling rate, in Hz

# DEFINE variable names to grab from .mat file
spikes_var_name = 'spikes_long'
lfp_var_name = 'lfp_long'

phase_var_names = []
for rhythm in rhythms:

    low = rhythm.split('_')[1]
    high = rhythm.split('_')[2]

    if cell_type == 'int':
        # phase_var_names.append(rhythm.split('_')[0] + 'phase_long') #for the ints
        phase_var_names.append(rhythm.split('_')[0] + '_' + str(low) + 'to' + str(high) + '_phase_long') #for the ints

    if cell_type == 'pyr':
        phase_var_names.append(rhythm + 'phase_' + str(low) + 'to' + str(high) + 'Hz_long')


#cross validation test set size
test_size = 0.5

#number of train test splits/folds for all models
n_desired_folds = 20


#decide whether you'll do complete models with logodds of both history and phase probs,
#rather than raw history data

rng = np.random.default_rng() #this is just a random number to make sure you don't overwrite previously saved results

logodds_completemdls = 1


if logodds_completemdls == 1:
    subfolder_name = cell_type + 'logoddsCompleteSinglePredMdls'+ str(rng.integers(low=0,high=1300)) + '/'
else:
    subfolder_name = cell_type + 'logoddsCompleteMdls' + str(rng.integers(low=0,high=1300)) + '/'


# PREALLOCATE VARS TO STORE POPULATION DATA
coeff_clong = []
coeff_TRANS_hlong = []
coeff_TRANS_hshort = []
coeff_cshort = []

kde_bandwidths = []
pop_losses_train = []
pop_losses_test = []




for n in tqdm(range(len(data_folders[58:]))):

    #reset this var for each neuron
    all_train_labels = []

    if cell_type == 'int':

        #uncomment below for old directory structure
        # nrn_path = os.path.join(data_path,data_folders[n])
        #
        # for file in os.listdir(nrn_path):
        #     if not file.startswith('.') & file.endswith('.mat'):
        #         file_name = file.split('.mat')[0]
        #         file_path = os.path.join(nrn_path, file_name)
        #
        # session_name = file_path + '_results'

        #the directory structure for high v is more similar to pyr cells structure
        file_path = os.path.join(data_path,data_folders[n])
        file_name = data_folders[n].split('.mat')[0]

        session_name = file_name + '_results'


    if cell_type == 'pyr':
        #for pyr cells, data_folders are actually files
        file_path = os.path.join(data_path,data_folders[n])
        file_name = data_folders[n].split('mat')[0]

        session_name = file_name + '_results'


    data = sio.loadmat(file_path,squeeze_me=1)

    if data['spikes_long'].sum() >= 50:

        for phase_var_name in phase_var_names:

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

            #keep only trials in df that contain at least one spike
            df = session.keep_trials_with_spikes()


            # generate the unique train test n_splits if you haven't done so
            # already for this neuron

            if bool(all_train_labels) == 0:

                #figure out how many trials the cell has, and then use the test size to determine how many training trials should be in each combination
                trial_labels = list(set(df['trial_labels'].to_list()))
                n_trials = len(trial_labels)
                trials_per_combo = round(n_trials * (1 - test_size))

                all_train_labels = train_trial_combinations_shuffle(trial_labels,trials_per_combo,n_desired_folds)

                N_FOLDS = len(all_train_labels)
                # constraint to randomly sample combinations up to 20, if there are >20 possible combinations
                if N_FOLDS > n_desired_folds:

                    all_train_labels = random.sample(all_train_labels, n_desired_folds)
                    N_FOLDS = len(all_train_labels)


            # frequency_edges = [low,high]

            for fold in tqdm(range(N_FOLDS),leave=False):
                # MAIN_SAVEPATH = 'RESULTS/PhaseHistModels_AllCells_pyr_byOdorPos/' + subfolder_name + rhythm + '_' + str(longhistlen) + 'ms' + '/folds/' + str(fold)
                NEURON_SAVEPATH = 'RESULTS/PHASE_HIST_MODELS_highvapproach_all_realcells/' + subfolder_name + phase_var_name.split('_')[0] + '_' + str(longhistlen) + 'ms' + '/folds/' + str(fold)


                current_train_labels = all_train_labels[fold]
                tmp = df[df['trial_labels'].isin(current_train_labels)]

                if loocv == 1:
                    kde_cvsplits = tmp['spikes'].sum() #setting up leave one out CV for bandwidth selection

                else:
                    kde_cvsplits = 5

                df, train, test, probs_df_test, probs_df_train, coeff_clong, coeff_hlong, coeff_hshort, coeff_cshort, coeff_TRANS_hshort, coeff_TRANS_hlong, kde_params, kde_bandwidth = kdePhase_logregHistory_models(df,data_folders[n],
                                                                                                                                                                                                                        current_train_labels,
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


                # df.to_csv(os.path.join(nrn_savepath,r'raw_data.csv'))
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
