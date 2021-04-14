# This function runs the following models:
# (1) phase models (kernel density estimators & Bayes rule)
# (2) short history models (logistic regression, spike history)
# (3) long history models (logistic regression, spike history, with L2 penalty)
# (4) complete short (logistic regression, spike history & log odds phase probs)
# (5) complete long (logistic regression, spike history & log odds phase probs, with L2 penalty)

import numpy as np
import pandas as pd

from crossval_fns import train_test_split_bytrials
from kde_spikephase import kde_spikephase_estimator
from mdl_eval_tools import bayes, logistic, logodds
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


def kdePhase_logregHistory_models(df,cell_id,test_size,kde_cvsplits,kde_gridsize,kde_narrow,kde_wide,kde_nbw,
                                    hist_long_col,hist_short_col,logodds_completemdls):
    """logodds_completemdls: boolean,
                            0: refit the history time lag coefficients for completes,
                            1: fit coeffs to logodds(history probs)"""

    n_trials = len(set(df['trial_labels'].to_list()))

    train, test = train_test_split_bytrials(df,test_size,n_trials)

    #compute the probability density function of phase given spiking in the training set
    pdf_train, kde_params, kde_bandwidth, x = kde_spikephase_estimator(train,
                                                    kde_cvsplits,
                                                    kde_gridsize,
                                                    kde_narrow,
                                                    kde_wide,
                                                    kde_nbw)

    #set up phase timeseries to predict over
    train = train.sort_index()
    train_list = np.array(train['phase'].values).reshape(-1,1)

    #compute the probability of phase given spiking over time
    prob_phase_given_spike_train = np.exp(kde_params.score_samples(train_list[:None]))
    prob_phase_given_spike_train = prob_phase_given_spike_train*3 #normalized for "wrapped gaussian" method

    #compute average spiking probability in train set
    spikes_prior_train = train['spikes'].sum()/train.shape[0]
    #compute the phase prior (uniform)
    phase_prior = 1/(2*np.pi)

    prob_spike_given_phase_train = bayes(prob_phase_given_spike_train,
                                            spikes_prior_train,
                                            phase_prior)

    logodds_phase_probs = logodds(prob_spike_given_phase_train)
    #unit test: ensure the log-odds recovers the probabilities
    if logistic(logodds_phase_probs).all() != prob_spike_given_phase_train.all():
        raise Exception('flawed computation: logistic(mu) must match logodds(p)')

    ##========================================================================
    ##                              TRAINING
    ##========================================================================
    y=[]
    y=train['spikes']

    #set up the regressors for logistic regression: long history
    regressor_hlong = train[hist_long_col].values
    #fit long history model (input: 250 ms of spike history)
    logReg_hlong = LogisticRegression(penalty='l2',fit_intercept=True,solver='lbfgs')
    logReg_hlong.fit(regressor_hlong,y)

    #set up the regressors for logistic regression: short history
    regressor_hshort = train[hist_short_col].values
    #fit short history model (input: 3 ms of spike history)
    logReg_hshort = LogisticRegression(penalty='none',fit_intercept=True,solver='lbfgs')
    logReg_hshort.fit(regressor_hshort,y)

    #set up the remaining regressor for logistic regression: complete long history
    regressor_p = logodds(prob_spike_given_phase_train)

    if logodds_completemdls == 0:
        #set up the full complete short regressor
        regressor_cshort = np.hstack((regressor_hshort,regressor_p.reshape(-1,1)))
        #fit complete short history model (input: 3 ms of spike history, )
        logReg_cshort = LogisticRegression(penalty='none',fit_intercept=True,solver='lbfgs')
        logReg_cshort.fit(regressor_cshort,y)

        #set up the full complete long regressor
        regressor_clong = np.hstack((regressor_hlong,regressor_p.reshape(-1,1)))
        #fit complete long history model (input: 250 ms of spike history, )
        logReg_clong = LogisticRegression(penalty='l2',fit_intercept=True,solver='lbfgs')
        logReg_clong.fit(regressor_clong,y)

    #==============================================================================
    # =======            TRAINING SET PREDICTIONS (OVERFIT))          =============
    #==============================================================================
    #phase probs needs no transformation
    probs_p_train = prob_spike_given_phase_train

    probs_hshort_train = logReg_hshort.predict_proba(regressor_hshort)[:,1]

    probs_hlong_train = logReg_hlong.predict_proba(regressor_hlong)[:,1]

    if logodds_completemdls == 0:

        probs_cshort_train = logReg_cshort.predict_proba(regressor_cshort)[:,1]

        probs_clong_train = logReg_clong.predict_proba(regressor_clong)[:,1]

    else:
        #fit the logodds short history model
        logReg_TRANS_hshort = LogisticRegression(penalty='none',fit_intercept=True,solver='lbfgs')
        logReg_TRANS_hshort.fit(logodds(probs_hshort_train).reshape(-1,1),y)

        #fit the logodds long history model
        logReg_TRANS_hlong = LogisticRegression(penalty='none',fit_intercept=True,solver='lbfgs')
        logReg_TRANS_hlong.fit(logodds(probs_hlong_train).reshape(-1,1),y)

        #set up the full complete short regressor
        regressor_cshort = np.hstack((logodds(probs_hshort_train).reshape(-1,1),regressor_p.reshape(-1,1)))
        #fit complete short history model (input: logodds of short history probs, )
        logReg_cshort = LogisticRegression(penalty='none',fit_intercept=True,solver='lbfgs')
        logReg_cshort.fit(regressor_cshort,y)

        #set up the full complete long regressor
        regressor_clong = np.hstack((logodds(probs_hlong_train).reshape(-1,1),regressor_p.reshape(-1,1)))
        #fit complete long history model (input: logodds of long history probs, )
        logReg_clong = LogisticRegression(penalty='none',fit_intercept=True,solver='lbfgs')
        logReg_clong.fit(regressor_clong,y)

        #generate predictions on the training set
        probs_TRANS_hshort_train = logReg_TRANS_hshort.predict_proba(logodds(probs_hshort_train).reshape(-1,1))[:,1]

        probs_TRANS_hlong_train = logReg_TRANS_hlong.predict_proba(logodds(probs_hlong_train).reshape(-1,1))[:,1]

        probs_clong_train = logReg_clong.predict_proba(regressor_clong)[:,1]

        probs_cshort_train = logReg_cshort.predict_proba(regressor_cshort)[:,1]


    if logodds_completemdls == 0:
        #collect the predictions
        probs_df_train = []
        probs_df_train = pd.DataFrame({'p_clong': probs_clong_train,
                                    'p_hlong': probs_hlong_train,
                                    'p_phase': probs_p_train,
                                    'p_hshort': probs_hshort_train,
                                    'p_cshort': probs_cshort_train})
    else:
        #collect the predictions
        probs_df_train = []
        probs_df_train = pd.DataFrame({'p_clong': probs_clong_train,
                                    'p_Transhlong': probs_TRANS_hlong_train,
                                    'p_hlong': probs_hlong_train,
                                    'p_phase': probs_p_train,
                                    'p_Transhshort': probs_TRANS_hshort_train,
                                    'p_hshort': probs_hshort_train,
                                    'p_cshort': probs_cshort_train})


    coeff_clong = {'c_clong': logReg_clong.coef_[0].tolist()}
    coeff_clong = pd.DataFrame(coeff_clong)
    coeff_clong = coeff_clong.T
    coeff_clong['intercept'] = logReg_clong.intercept_[0]
    coeff_clong['cell id'] = cell_id


    coeff_hlong = {'c_hlong': logReg_hlong.coef_[0].tolist()}
    coeff_hlong = pd.DataFrame(coeff_hlong)
    coeff_hlong = coeff_hlong.T
    coeff_hlong['intercept'] = logReg_hlong.intercept_[0]
    coeff_hlong['cell id'] = cell_id


    coeff_hshort = {'c_hshort': logReg_hshort.coef_[0].tolist()}
    coeff_hshort = pd.DataFrame(coeff_hshort)
    coeff_hshort = coeff_hshort.T
    coeff_hshort['intercept'] = logReg_hshort.intercept_[0]
    coeff_hshort['cell id'] = cell_id


    coeff_cshort = {'c_cshort': logReg_cshort.coef_[0].tolist()}
    coeff_cshort = pd.DataFrame(coeff_cshort)
    coeff_cshort = coeff_cshort.T
    coeff_cshort['intercept'] = logReg_cshort.intercept_[0]
    coeff_cshort['cell id'] = cell_id

    coeff_TRANS_hshort = []
    coeff_TRANS_hlong = []

    if logodds_completemdls == 1:

        coeff_TRANS_hshort = {'c_hshort': logReg_TRANS_hshort.coef_[0].tolist()}
        coeff_TRANS_hshort = pd.DataFrame(coeff_TRANS_hshort)
        coeff_TRANS_hshort = coeff_TRANS_hshort.T
        coeff_TRANS_hshort['intercept'] = logReg_TRANS_hshort.intercept_[0]
        coeff_TRANS_hshort['cell id'] = cell_id

        coeff_TRANS_hlong = {'c_hlong': logReg_TRANS_hlong.coef_[0].tolist()}
        coeff_TRANS_hlong = pd.DataFrame(coeff_TRANS_hlong)
        coeff_TRANS_hlong = coeff_TRANS_hlong.T
        coeff_TRANS_hlong['intercept'] = logReg_TRANS_hlong.intercept_[0]
        coeff_TRANS_hlong['cell id'] = cell_id


    ##========================================================================
    ##                    TEST (HELDOUT) SET PREDICTIONS
    ##========================================================================
    test = test.sort_index()
    test_list = np.array(test['phase'].values)

    probs_phase_given_spike_test = np.exp(kde_params.score_samples(test_list[:,None]))
    probs_phase_given_spike_test = probs_phase_given_spike_test*3


    #compute the probability of phase given spiking over time
    #as spikes prior, use the average prob of spiking in **train set**
    prob_spike_given_phase_test = bayes(probs_phase_given_spike_test,
                                            spikes_prior_train,
                                            phase_prior)

    logodds_phase_probs = logodds(prob_spike_given_phase_test)
    #unit test: ensure the log-odds can be re-converted to the original probabilities
    if logistic(logodds_phase_probs).all() != prob_spike_given_phase_test.all():
        raise Exception('flawed computation: logistic(mu) must match logodds(p)')

    ##========================================================================
    ##                    SET UP TEST DATA TO PREDICT OVER
    ##========================================================================
    heldout_hlong = test[hist_long_col].values
    heldout_hshort = test[hist_short_col].values

    #model predictions for heldout set
    probs_p_test = prob_spike_given_phase_test

    probs_hlong_test = logReg_hlong.predict_proba(heldout_hlong)[:,1]

    probs_hshort_test = logReg_hshort.predict_proba(heldout_hshort)[:,1]

    if logodds_completemdls == 0:
        #set up heldout data for complete short model
        heldout_cshort = np.hstack((heldout_hshort,logodds(probs_p_test).reshape(-1,1)))
        #setup heldout data for complete long
        heldout_clong = np.hstack((heldout_hlong,logodds(probs_p_test).reshape(-1,1)))

        #make complete model predictions
        probs_cshort_test = logReg_cshort.predict_proba(heldout_cshort)[:,1]

        probs_clong_test = logReg_clong.predict_proba(heldout_clong)[:,1]

    else:
        #predict probs from logodds short history model
        probs_TRANS_hshort_test = logReg_TRANS_hshort.predict_proba(logodds(probs_hshort_test).reshape(-1,1))[:,1]

        #predict probs from logodds long history model
        probs_TRANS_hlong_test = logReg_TRANS_hlong.predict_proba(logodds(probs_hlong_test).reshape(-1,1))[:,1]

        #set up heldout data for complete short model
        heldout_cshort = np.hstack((logodds(probs_hshort_test).reshape(-1,1),logodds(probs_p_test).reshape(-1,1)))
        #setup heldout data for complete long
        heldout_clong = np.hstack((logodds(probs_hlong_test).reshape(-1,1),logodds(probs_p_test).reshape(-1,1)))

        #make complete model predictions
        probs_cshort_test = logReg_cshort.predict_proba(heldout_cshort)[:,1]

        probs_clong_test = logReg_clong.predict_proba(heldout_clong)[:,1]

    if logodds_completemdls == 0:
        probs_df_test = []
        probs_df_test = pd.DataFrame({'p_clong': probs_clong_test,
                                    'p_hlong': probs_hlong_test,
                                    'p_phase': probs_p_test,
                                    'p_hshort': probs_hshort_test,
                                    'p_cshort': probs_cshort_test})

    else:
        probs_df_test = []
        probs_df_test = pd.DataFrame({'p_clong': probs_clong_test,
                                    'p_Transhlong': probs_TRANS_hlong_test,
                                    'p_hlong': probs_hlong_test,
                                    'p_phase': probs_p_test,
                                    'p_Transhshort': probs_TRANS_hshort_test,
                                    'p_hshort': probs_hshort_test,
                                    'p_cshort': probs_cshort_test})


    return df, train, test, probs_df_test, probs_df_train, coeff_clong, coeff_hlong, coeff_hshort, coeff_cshort, coeff_TRANS_hshort, coeff_TRANS_hlong,kde_params,kde_bandwidth
