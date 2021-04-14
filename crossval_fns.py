
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

def train_test_split_bytrials(df,test_size,n_trials):
    """Partitions the dataset into train and test splits by trial label, that is,
    keeping all consecutive trial timestamps together"""

    n_trials_test = round(n_trials*test_size)
    n_trials_train = n_trials - n_trials_test

    all_labels = np.arange(n_trials)

    #sample without replacement
    train_labels = random.sample(list(all_labels), n_trials_train)

    train = df[df['trial_labels'].isin(train_labels)]
    test = df[~df['trial_labels'].isin(train_labels)]

    return train, test


def train_test_split_chrono(test_size,n_trials):
    """Partitions the dataset into train and test splits; assuming that the
    data is trial-based, the train set consists of the first trials, and the
    test consists of the last trials"""

    n_trials_test = round(n_trials*test_size)
    n_trials_train = n_trials - n_trials_test

    return n_trials_train, n_trials_test

def run_train_splits_df(df,n_splits,max_poly_deg,n_bins):
    """Partitions a train subset of df data, performs polynomial regression
    for various degrees, up to max_poly_deg, and ouputs a dataframe containing
    degree | logloss | split number """

    d = []
    models = []
    poly_fit_init = []

    for spl in tqdm(range(n_splits)):

        ### Split your data into train and validation sets
        train, val = train_test_split(df,test_size=.5)

        ### Compute crude p(phase_bin | spk=1) and prep vars for regressions
        spks_by_phase = []
        for y, x in zip(train['spikes'],train['phase']):
            if y == 1:
                spks_by_phase.append(x)
        # plt.figure()
        # plt.hist(spks_by_phase,n_bins)
        # plt.show()

        crude_conditional, binedges = np.histogram(spks_by_phase,bins=n_bins)
        conditional_trn = crude_conditional / len(spks_by_phase) #prob of phase given spk

        train_set_len = train.shape[0]

        for deg in range(max_poly_deg):

            ## Transform phase bin values into high-d, polynomial feature space
            # representation
            polynomial_reg = PolynomialFeatures(degree=deg)
            polynom_bins = polynomial_reg.fit_transform(binedges[:-1].reshape(-1,1))

            ## Apply linear regression between transformed phase bins and
            #p(phase_bin | spk=1)

            lin_reg = LinearRegression()
            lin_reg.fit(polynom_bins,conditional_trn)

            ## Train logistic regression on the relationship between the predicted
            # p(phase | spk=1) learned from training phases and the actually
            # observed spikes in the spikes training set

            polynom_x_train = polynomial_reg.fit_transform(train['phase'].values.reshape(-1,1))
            conditional_pred_train = lin_reg.predict(polynom_x_train)

            log_reg = LogisticRegression()
            conditional_pred_train = conditional_pred_train.reshape(-1,1)
            log_reg.fit(conditional_pred_train,train['spikes'])

            ## Now extend trained models to validation sets

            ## Get polynomial features and p(phase | spk=1) for the validation set

            polynom_x_val = polynomial_reg.transform(val['phase'].values.reshape(-1,1))
            conditional_pred_val = lin_reg.predict(polynom_x_val)

            # Plug p(phase | spk=1) into trained logistic regression model to get log loss
            spikes_predict_val = log_reg.predict_proba(conditional_pred_val.reshape(-1,1))

            ## How well does the trained logistic regression model predict spikes
            # for the validation set?

            losses = log_loss(val['spikes'],spikes_predict_val)
            poly_mse = mean_squared_error(val['spikes'],conditional_pred_val)

            ## Aggregate results in a data frame
            d.append({'poly deg': deg,
                    'log loss': losses,
                    'rmse': np.sqrt(poly_mse),
                    'data_split': spl})

            loglosses_poly_df = pd.DataFrame(data=d)

            models.append(lin_reg)
            poly_fit_init.append(polynomial_reg)

    return loglosses_poly_df, models, poly_fit_init, train_set_len


def run_train_splits(x_subset,y_subset,n_splits,max_poly_deg,n_bins):
    """Partitions a train subset of data, performs polynomial regression
    for various degrees, up to max_poly_deg, and ouputs a dataframe containing
    degree | logloss | split number """


    d = []

    for spl in range(n_splits):

        ### Split your data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(x_subset,
                                                y_subset,
                                                test_size=0.5)

        ### Compute crude p(phase_bin | spk=1) and prep variables for regressions

        xgy1 = []
        for y,x in zip(y_train,X_train):
            if y==1:
                xgy1.append(x)
        plt.figure()
        plt.hist(xgy1,n_bins)
        plt.show()

        freq_trn, binedges = np.histogram(xgy1,bins = n_bins)
        pxgy1_trn = freq_trn / len(xgy1) #Prob of X Given Y = 1

        models = []
        poly_fit_init = []
        print(len(poly_fit_init))


        ### Polynomial Regression
        for deg in range(max_poly_deg):

            ## Transform phase bin values into high-d, polynomial feature space
             # representation

            polynomial_reg = PolynomialFeatures(degree=deg)
            polynom_bins = polynomial_reg.fit_transform(binedges[:-1].reshape(-1,1))

            ## Apply linear regression between transformed phase bins and
             #p(phase_bin | spk=1)

            lin_reg = LinearRegression()
            lin_reg.fit(polynom_bins,pxgy1_trn)

            ## Train logistic regression on the relationship between the predicted
             # p(phase | spk=1) learned from training phases and the actually
             # observed spikes in the spikes training set

            polynom_x_train = polynomial_reg.fit_transform(X_train.reshape(-1,1))
            pxgy1_pred_train = lin_reg.predict(polynom_x_train)

            log_reg = LogisticRegression()
            pxgy1_pred_train = pxgy1_pred_train.reshape(-1,1)
            log_reg.fit(pxgy1_pred_train,y_train)

            ## Now extend trained models to validation sets

            ## Get polynomial features and p(phase | spk=1) for the validation set

            polynom_x_val = polynomial_reg.transform(X_val.reshape(-1,1))
            pxgy1_pred_val = lin_reg.predict(polynom_x_val)

            # Plug p(phase | spk=1) into trained logistic regression model
            y_predict_val = log_reg.predict_proba(pxgy1_pred_val.reshape(-1,1))

            ## How well does the trained logistic regression model predict spikes
             # for the validation set?

            losses = log_loss(y_val,y_predict_val)

            ## Aggregate results in a data frame
            d.append({'poly deg': deg+1,
                    'log loss': losses,
                    'data_split': spl})

            loglosses_poly_df = pd.DataFrame(data=d)

            models.append(lin_reg)
            poly_fit_init.append(polynomial_reg)

    return loglosses_poly_df, models, poly_fit_init
