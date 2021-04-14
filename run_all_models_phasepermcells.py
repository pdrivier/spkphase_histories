#final version code for the bootstrapped cells
#this code borrows from the OG code, so it is less clean and modular than the
#newest versions of the kdephase-logreghistory model fitting code
#changes to the OG code are:
#(1) phase test probs are now computed using the training set spikes prior, to
#be actually comparable to the history models
#(2) the penalties are explicitly set to "none" for the complete models




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
from mdl_eval_tools import bayes, ksPlot, logloss, logodds, logistic
from process_trial_data import find_true_trial_end_samples
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

# LOAD DATA:grab from previous runs with real cells, so you can keep the training trials constant
rhythm = 'theta'
longhistlen = 250

# SET BANDWITH RANGE FOR KERNEL DENSITY ESTIMATION
## Select the bandwidth range for kernel
NARROW = 1/30
WIDE = 1/0.6
N_BW = 20

N_FOLDS = np.arange(5,6)
N_PERMS = 100

for fold in N_FOLDS:

    traindata_path = 'FINAL/Entropy2020PhasePermDataset/' + 'logoddsCompleteSinglePredMdls1061/' +  rhythm + str(longhistlen) + 'ms' + '/folds/' + str(fold) +  '/single_neuron/'

    traindata_folders=[]
    traindata_folders = [f for f in os.listdir(traindata_path) if not f.startswith('.')]
    traindata_folders = sorted(traindata_folders)

    testdata_path = 'FINAL/Entropy2020_Fig4_Dataset/' + 'logoddsCompleteSinglePredMdls1061/' + rhythm + str(longhistlen) + 'ms' + '/folds/' + str(fold) + '/single_neuron/'

    testdata_folders=[]
    testdata_folders = [f for f in os.listdir(testdata_path) if not f.startswith('.')]
    testdata_folders = sorted(testdata_folders)

    MAIN_SAVEPATH = 'FINAL/Entropy2020PhaseHistModels_PhasePermCells/'  + 'logoddsCompleteSinglePredMdls1061/' + rhythm + str(longhistlen) + 'ms' + '/folds/' + str(fold)
    NEURON_SAVEPATH = MAIN_SAVEPATH + '/single_neuron/'
    POPULATION_SAVEPATH = MAIN_SAVEPATH + '/population/'


    hist_longest = 250
    histlong = 250
    hist_long_col = list(reversed(range(hist_longest-histlong,hist_longest)))
    hist_long_col = [str(i) for i in hist_long_col]

    histshort = 3
    hist_short_col = list(reversed(range(hist_longest-histshort,
                                             hist_longest)))
    hist_short_col = [str(i) for i in hist_short_col]



    # PREALLOCATE VARS TO STORE POPULATION DATA
    permphase_coeff_clong = []
    permphase_coeff_cshort = []

    kde_bandwidths = []
    permphase_pop_losses_train = []
    permphase_pop_losses_test = []

    for n in tqdm(range(len(traindata_folders))):#   np.arange(100,101)

        trainnrn_path = os.path.join(traindata_path,traindata_folders[n])
        train_data = pd.read_csv(os.path.join(trainnrn_path,'permphase.csv'))

        testnrn_path = os.path.join(testdata_path,testdata_folders[n])
        test_data = pd.read_csv(os.path.join(testnrn_path,'test_data.csv'))
        probs_models_OGtrain = pd.read_csv(os.path.join(testnrn_path,'probs_models_train.csv'))
        probs_models_OGtest = pd.read_csv(os.path.join(testnrn_path,'probs_models_test.csv'))


    #===============================================================================
    # ----------------> KDE: COMPUTE PERM PHASE MODEL ESTIMATES <-------------------
    #===============================================================================
        permphase_probs_df_train = []
        permphase_probs_df_test = []

        for perm in range(N_PERMS):
            permphase_col = 'phaseperm' + str(perm)

            train_spk_phase=[]
            for s,p in zip(train_data['spikes'].values,train_data[permphase_col].values):
                if s==1:
                    train_spk_phase.append(p)

            train_spk_phase = np.array(train_spk_phase)
            phase_grid = np.linspace(-np.pi,np.pi,1000)
            n_splits = 5  #cross validation folds

            grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                                {'bandwidth': np.linspace(NARROW,WIDE,N_BW)},
                                cv=n_splits) #n-split-fold cross validation

            #hacky but effective way to get the kernel to treat phase circularly
            cycle1 = train_spk_phase - 2*np.pi
            cycle2 = train_spk_phase
            cycle3 = train_spk_phase + 2*np.pi

            fitting_sample = np.hstack((cycle1,cycle2,cycle3))
            grid.fit(fitting_sample[:,None])
            # print(grid.best_params_)
            kde_bandwidths.append(grid.best_params_)

            kde = grid.best_estimator_
            pdf = np.exp(kde.score_samples(phase_grid[:, None]))
            pdf = pdf * 3 #normalize for the extra data added on either side

            # set up p(phase | spike) for the full train dataset
            train = train_data.sort_index()
            train_list = np.array(train_data[permphase_col].values)

            conditional_pred_train = np.exp(kde.score_samples(train_list[:,None]))
            conditional_pred_train = conditional_pred_train*3 # normalize

            ### obtain p(spk|phase)
            spikes_prior_train = train_data['spikes'].sum()/train_data.shape[0] #n time samples s == 1/ n timesamples total
            phase_prior = 1/(2*np.pi) #uniform prior over phases

            spike_phase_posterior_train = bayes(conditional_pred_train,
                                                spikes_prior_train,phase_prior)

            # derive log odds to serve as regressor instead of raw probs
            mu_train = logodds(spike_phase_posterior_train)

            #unit test: ensure mu_train recovers the probabilities
            if logistic(mu_train).all() != spike_phase_posterior_train.all():
                raise Exception('flawed computation: logistic(mu) must match logodds(p)')


            # FIT models
            y=[]
            y = train_data['spikes']

            ###BUILD regressors for each model
            regressor_p = mu_train

            regressor_hlong = train_data[hist_long_col].values

            regressor_hshort = train_data[hist_short_col].values


            #===========================================================================
            #==============OBTAIN model performance on the TRAIN set====================
            #===========================================================================

            # preallocate model coefficients and probability vectors
            coeffs_clong = np.zeros((1,2))
            probs_clong_train = np.zeros((1,train.shape[0]))

            coeffs_cshort = np.zeros((1,2))
            probs_cshort_train = np.zeros((1,train.shape[0]))


            #grab the history model probabilities from the corresponding original model run
            probs_hlong_train = np.array(probs_models_OGtrain['p_hlong'].to_list())
            probs_hshort_train = np.array(probs_models_OGtrain['p_hshort'].to_list())


            regressor_clong = np.hstack((logodds(probs_hlong_train).reshape(-1,1),regressor_p.reshape(-1,1)))
            regressor_cshort = np.hstack((logodds(probs_hshort_train).reshape(-1,1),regressor_p.reshape(-1,1)))

            #FIT complete models
            x_c_ln = regressor_clong
            logReg_clong = LogisticRegression(penalty='none',fit_intercept=True,solver='lbfgs')
            logReg_clong.fit(x_c_ln,y)

            x_c_sh = regressor_cshort
            logReg_cshort = LogisticRegression(penalty='none',fit_intercept=True,solver='lbfgs')
            logReg_cshort.fit(x_c_sh,y)


            #Get model predictions for training set (to compare test against)
            probs_clong_train = logReg_clong.predict_proba(regressor_clong)[:,1]

            probs_p_train = spike_phase_posterior_train

            probs_cshort_train = logReg_cshort.predict_proba(regressor_cshort)[:,1]


            train_tmpdict = {'p_clong': probs_clong_train,
                                        'p_hlong': probs_hlong_train,
                                        'p_phase': probs_p_train,
                                        'p_hshort': probs_hshort_train,
                                        'p_cshort': probs_cshort_train,
                                        'p_ph_spk': conditional_pred_train,
                                        'log_odds': mu_train,
                                         'perm_id': np.tile(perm,len(probs_clong_train))}
            if perm == 0:
                permphase_probs_df_train = pd.DataFrame(train_tmpdict)
            else:
                train_tmpdf = pd.DataFrame(train_tmpdict)
                permphase_probs_df_train = pd.concat([permphase_probs_df_train,train_tmpdf],axis=0)


            ### Compute log losses on the train set
            y=[]
            y = train_data['spikes']

            avg_loss_clong_train, _, _ = logloss(probs_clong_train,y)
            avg_loss_hlong_train, _, _ = logloss(probs_hlong_train,y)
            avg_loss_p_train, _, _ = logloss(probs_p_train,y)
            avg_loss_hshort_train, _, _ = logloss(probs_hshort_train,y)
            avg_loss_cshort_train, _, _ = logloss(probs_cshort_train,y)

            permphase_pop_losses_train.append({'cell id': traindata_folders[n],
                                     'log loss complete_long mdl': avg_loss_clong_train,
                                     'log loss history_long mdl': avg_loss_hlong_train,
                                     'log loss phase mdl': avg_loss_p_train,
                                     'log loss history_short mdl': avg_loss_hshort_train,
                                     'log loss complete_short mdl': avg_loss_cshort_train,
                                     'perm_id': perm})

            permphase_coeff_clong.append({'cell id': traindata_folders[n],
                                    'c_clong': logReg_clong.coef_[0],
                                         'perm_id': perm})

            permphase_coeff_cshort.append({'cell id': traindata_folders[n],
                                    'c_cshort': logReg_cshort.coef_[0],
                                          'perm_id': perm})


            #===========================================================================
            #==============OBTAIN model predictions for the TEST set====================
            #===========================================================================
            test = test_data.sort_index()
            test_list = np.array(test_data['phase'].values)

            conditional_pred_test = np.exp(kde.score_samples(test_list[:,None]))
            conditional_pred_test = conditional_pred_test*3

            # spikes_prior_test = test_data['spikes'].sum()/test_data.shape[0]

            ## obtain p(spk | phase) using training spikes prior (to make) probs comparable to history model probs
            spike_phase_posterior_test = bayes(conditional_pred_test,spikes_prior_train,phase_prior)

            # derive log odds to serve as regressor instead of raw probs for completes
            mu_test = logodds(spike_phase_posterior_test)

            #unit test: to ensure mu_test is correctly computed
            if logistic(mu_test).all() != spike_phase_posterior_test.all():
                raise Exception('flawed computation: logistic(mu) must match logodds(p)')

            #===============================================================================
            #========================BUILD NEW PREDICTORS===================================
            #===============================================================================
            new_p = mu_test

            new_hlong = test_data[hist_long_col].values

            new_hshort = test_data[hist_short_col].values

            # preallocate model coefficients and probability vectors

            probs_clong_test = np.zeros((1,test.shape[0]))

            probs_cshort_test = np.zeros((1,test.shape[0]))

            #grab the history model probabilities from the corresponding original model run
            probs_hlong_test = np.array(probs_models_OGtest['p_hlong'].to_list())
            probs_hshort_test = np.array(probs_models_OGtest['p_hshort'].to_list())


            new_clong = np.hstack((logodds(probs_hlong_test).reshape(-1,1),new_p.reshape(-1,1)))
            new_cshort = np.hstack((logodds(probs_hshort_test).reshape(-1,1),new_p.reshape(-1,1)))

            #Get model predictions for test set
            probs_clong_test = logReg_clong.predict_proba(new_clong)[:,1]

            probs_p_test = spike_phase_posterior_test

            probs_cshort_test = logReg_cshort.predict_proba(new_cshort)[:,1]

            test_tmpdict = {'p_clong': probs_clong_test,
                                        'p_hlong': probs_hlong_test,
                                        'p_phase': probs_p_test,
                                        'p_hshort': probs_hshort_test,
                                        'p_cshort': probs_cshort_test,
                                        'p_ph_spk': conditional_pred_test,
                                        'log_odds': mu_test,
                                         'perm_id': np.tile(perm,len(probs_clong_test))}
            if perm == 0:
                permphase_probs_df_test = pd.DataFrame(test_tmpdict)
            else:
                test_tmpdf = pd.DataFrame(test_tmpdict)
                permphase_probs_df_test = pd.concat([permphase_probs_df_test,test_tmpdf],axis=0)


             ### Compute log losses for test set
            y=[]
            y = test_data['spikes']

            avg_loss_clong_test, _, _ = logloss(probs_clong_test,y)
            avg_loss_hlong_test, _, _ = logloss(probs_hlong_test,y)
            avg_loss_p_test, _, _ = logloss(probs_p_test,y)
            avg_loss_hshort_test, _, _ = logloss(probs_hshort_test,y)
            avg_loss_cshort_test, _, _ = logloss(probs_cshort_test,y)

            permphase_pop_losses_test.append({'cell id': traindata_folders[n],
                                'log loss complete_long mdl': avg_loss_clong_test,
                                'log loss history_long mdl': avg_loss_hlong_test,
                                'log loss phase mdl': avg_loss_p_test,
                                'log loss history_short mdl': avg_loss_hshort_test,
                                'log loss complete_short mdl': avg_loss_cshort_test,
                                   'perm_id': perm})

        #===============================================================================
        #===========create the save folder if it doesn't yet exist======================
        #===============================================================================

        if not os.path.exists(NEURON_SAVEPATH):
            os.makedirs(NEURON_SAVEPATH)

        nrn_savepath = os.path.join(NEURON_SAVEPATH,traindata_folders[n])
        if not os.path.exists(nrn_savepath):
            os.makedirs(nrn_savepath)


        permphase_probs_df_train.to_csv(os.path.join(nrn_savepath,r'permphase_probs_models_train.csv'))
        permphase_probs_df_test.to_csv(os.path.join(nrn_savepath,r'permphase_probs_models_test.csv'))

    permphase_pop_losses_train = pd.DataFrame(data=permphase_pop_losses_train)
    permphase_pop_losses_test = pd.DataFrame(data=permphase_pop_losses_test)


    col=[]
    # titles_clong = list(reversed(np.arange(0,hist_longlag)))
    coeff_clong_df = pd.DataFrame(data=permphase_coeff_clong)
    col = pd.DataFrame(coeff_clong_df['c_clong'].tolist())
    coeff_clong_df = coeff_clong_df.drop(['c_clong'],axis=1)
    permphase_coeff_clong_df = coeff_clong_df.join(col)

    col=[]
    # titles_cshort = [list(reversed(np.arange(0,hist_shortlag))),'phase']
    coeff_cshort_df = pd.DataFrame(data=permphase_coeff_cshort)
    col = pd.DataFrame(coeff_cshort_df['c_cshort'].tolist())
    coeff_cshort_df = coeff_cshort_df.drop(['c_cshort'],axis=1)
    permphase_coeff_cshort_df = coeff_cshort_df.join(col)

    #momentarily get rid of the cell_id column, so you can do the normalization
    f_trn = permphase_pop_losses_train[['log loss complete_long mdl',
                    'log loss history_long mdl',
                    'log loss phase mdl',
                    'log loss history_short mdl',
                    'log loss complete_short mdl']]

    tmp = f_trn.mean(axis=1)
    permphase_pop_losses_train = permphase_pop_losses_train.merge(tmp.rename('mean losses'),left_index=True,right_index=True)
    permphase_pop_losses_train_normed = f_trn.sub(permphase_pop_losses_train['mean losses'],axis=0)
    df_melted = pd.melt(permphase_pop_losses_train_normed)

    f_trn_n = permphase_pop_losses_train_normed[['log loss complete_long mdl',
                    'log loss history_long mdl',
                    'log loss phase mdl',
                    'log loss history_short mdl',
                    'log loss complete_short mdl']]
    sns.heatmap(f_trn_n.sort_values(by='log loss phase mdl'),annot=False, fmt="g", cmap="RdGy",center=0)
    plt.show()
    plt.close()

    f_tst = permphase_pop_losses_test[['log loss complete_long mdl',
                    'log loss history_long mdl',
                    'log loss phase mdl',
                    'log loss history_short mdl',
                    'log loss complete_short mdl']]

    tmp = f_tst.mean(axis=1)
    permphase_pop_losses_test = permphase_pop_losses_test.merge(tmp.rename('mean losses'),left_index=True,right_index=True)
    permphase_pop_losses_test_normed = f_tst.sub(permphase_pop_losses_test['mean losses'],axis=0)
    df_melted = pd.melt(permphase_pop_losses_test_normed)

    f_tst_n = permphase_pop_losses_test_normed[['log loss complete_long mdl',
                    'log loss history_long mdl',
                    'log loss phase mdl',
                    'log loss history_short mdl',
                    'log loss complete_short mdl']]
    sns.heatmap(f_tst_n.sort_values(by='log loss phase mdl'),annot=False, fmt="g", cmap="RdGy",center=0)
    plt.show()
    plt.close()

    sns.boxplot(data = df_melted, x = "variable", y="value")
    plt.show()
    plt.close()

    #### Save outputs: bring to jupyter notebook
    if not os.path.exists(POPULATION_SAVEPATH):
        os.makedirs(POPULATION_SAVEPATH)

    permphase_pop_losses_train_normed['cell id'] = permphase_pop_losses_train['cell id']
    permphase_pop_losses_train_normed['perm_id'] = permphase_pop_losses_train['perm_id']
    permphase_pop_losses_train_normed.to_csv(os.path.join(POPULATION_SAVEPATH,r'permphase_pop_losses_train_normed.csv'))
    permphase_pop_losses_train.to_csv(os.path.join(POPULATION_SAVEPATH,r'permphase_pop_losses_train.csv'))

    permphase_pop_losses_test_normed['cell id'] = permphase_pop_losses_test['cell id']
    permphase_pop_losses_test_normed['perm_id'] = permphase_pop_losses_test['perm_id']
    permphase_pop_losses_test_normed.to_csv(os.path.join(POPULATION_SAVEPATH,r'permphase_pop_losses_test_normed.csv'))
    permphase_pop_losses_test.to_csv(os.path.join(POPULATION_SAVEPATH,r'permphase_pop_losses_test.csv'))

    permphase_coeff_clong_df.to_csv(os.path.join(POPULATION_SAVEPATH,r'permphase_coeff_clong.csv'))
    permphase_coeff_cshort_df.to_csv(os.path.join(POPULATION_SAVEPATH,r'permphase_coeff_cshort.csv'))

    permphase_kde_bandwidths = pd.DataFrame(data=kde_bandwidths)
    permphase_kde_bandwidths.to_csv(os.path.join(POPULATION_SAVEPATH,r'permphase_kde_bandwidths.csv'))
