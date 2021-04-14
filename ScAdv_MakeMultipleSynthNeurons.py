#Make multiple synthetic neurons of each category:
#(1) no refractory + no phase
#(2) refractory + no phase
#(3) no refractory + phase
#(4) refractory + phase

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
import vonMises.vonMises as VM

from scipy import stats
from sim_data_tools import sim_binom_spktrn, make_phase_timeseries, Neuron
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from scipy.special import i0

def vonmises_pr(x,mu,kappa):
    """Simulates a vonmises distribution"""

    vm = np.exp(kappa*np.cos(x-mu))/(2*np.pi*i0(kappa))

    return vm

N_NEURONS = 50 #number of unique neurons to simulate in each category

TARGET_FR = 6  #actual firing rate will always slightly undershoot this
N_TRIALS = 48   #number of trials in the simulated session
LEN_TRIAL = 1500#in ms, duration of single trial
FREQ_SEQ = [8,8]#frequency to make phase timeseries for (if you change the
                #number in the vector, will get half trial one frequency,
                #half trial will be the other frequency)
MU = [np.pi, np.pi]#radians, peak location of spike-phase distribution
FS = 1000       #in Hz, sampling rate


for n in tqdm(range(N_NEURONS)):
    DATASAVEPATH = 'ScAdv_sim_neuron_data/' + 'sim' + str(n) +'/target_frate/' + str(TARGET_FR) + '/'

    if not os.path.exists(DATASAVEPATH):
        os.makedirs(DATASAVEPATH)

    if not os.path.exists(os.path.join(DATASAVEPATH,'ScAdv_sim_params_all/')):
        os.makedirs(os.path.join(DATASAVEPATH,'ScAdv_sim_params_all/'))
    #=========----------------------------==========================================
    #=========  NO REFRACTORY + NO PHASE  ------------------------------------------
    #=========----------------------------==========================================
    kappa = [0.0001,0.0001] #uniform over all phase
    has_refractory = 0 #no refractory period
    has_rhythm = 0 #no phase

    noRenoPh = Neuron(TARGET_FR, N_TRIALS, has_rhythm, has_refractory, FREQ_SEQ, MU, kappa, FS, LEN_TRIAL)

    phase_singletrial, mus_single_trial, kappas_single_trial, true_len_trial = noRenoPh.make_singletrial_phase()

    noRenoPh.make_multitrial_phase()
    df = noRenoPh.make_spike_probs()
    noRenoPh.label_trials()

    noRenoPh.df.to_csv(os.path.join(DATASAVEPATH,r'sim_noRefractorynoPhase_df.csv'))
    actual_frate = (df['spikes'].sum()/df.shape[0])*FS

    print('sim params: \n',
          'neuron id:' 'noRenoPh',
          'avg_frate: ',actual_frate,'\n',
          'n_trials: ',N_TRIALS,'\n',
          'freq_seq: ',FREQ_SEQ,'\n',
          'mu: ',MU,'\n',
          'kappa: ',kappa,'\n',
          'fs:',FS,'\n',
          '\n',file=open(os.path.join(DATASAVEPATH,'ScAdv_sim_params_all/',"noRenoPh_sim_params.txt"), "a"))

    #=========----------------------------==========================================
    #=========     REFRACTORY + NO PHASE  ------------------------------------------
    #=========----------------------------==========================================
    kappa = [0.0001,0.0001] #uniform over all phase
    has_refractory = 1 # has a refractory period
    has_rhythm = 0 #no phase

    RenoPh = Neuron(TARGET_FR, N_TRIALS, has_rhythm, has_refractory, FREQ_SEQ, MU, kappa, FS, LEN_TRIAL)

    phase_singletrial, mus_single_trial, kappas_single_trial, true_len_trial = RenoPh.make_singletrial_phase()

    RenoPh.make_multitrial_phase()
    df = RenoPh.make_spike_probs()
    RenoPh.label_trials()

    RenoPh.df.to_csv(os.path.join(DATASAVEPATH,r'sim_RefractorynoPhase_df.csv'))
    actual_frate = (df['spikes'].sum()/df.shape[0])*FS

    print('sim params: \n',
          'neuron id:' 'RenoPh',
          'avg_frate: ',actual_frate,'\n',
          'n_trials: ',N_TRIALS,'\n',
          'freq_seq: ',FREQ_SEQ,'\n',
          'mu: ',MU,'\n',
          'kappa: ',kappa,'\n',
          'fs:',FS,'\n',
          '\n',file=open(os.path.join(DATASAVEPATH,'ScAdv_sim_params_all/',"RenoPh_sim_params.txt"), "a"))

    #=========----------------------------==========================================
    #=========  NO  REFRACTORY +   PHASE  ------------------------------------------
    #=========----------------------------==========================================
    kappa = [2,2] #phase preference
    has_refractory = 0 # has a refractory period
    has_rhythm = 1 #rhythmic

    noRePh = Neuron(TARGET_FR, N_TRIALS, has_rhythm, has_refractory, FREQ_SEQ, MU, kappa, FS, LEN_TRIAL)

    phase_singletrial, mus_single_trial, kappas_single_trial, true_len_trial = noRePh.make_singletrial_phase()

    noRePh.make_multitrial_phase()
    df = noRePh.make_spike_probs()
    noRePh.label_trials()

    noRePh.df.to_csv(os.path.join(DATASAVEPATH,r'sim_noRefractoryPhase_df.csv'))
    actual_frate = (df['spikes'].sum()/df.shape[0])*FS

    print('sim params: \n',
          'neuron id:' 'noRePh',
          'avg_frate: ',actual_frate,'\n',
          'n_trials: ',N_TRIALS,'\n',
          'freq_seq: ',FREQ_SEQ,'\n',
          'mu: ',MU,'\n',
          'kappa: ',kappa,'\n',
          'fs:',FS,'\n',
          '\n',file=open(os.path.join(DATASAVEPATH,'ScAdv_sim_params_all/',"noRePh_sim_params.txt"), "a"))

    #=========----------------------------==========================================
    #=========    REFRACTORY +   PHASE    ------------------------------------------
    #=========----------------------------==========================================
    kappa = [2,2] # phase preference
    has_refractory = 1 # has a refractory period
    has_rhythm = 1 #rhythmic

    RePh = Neuron(TARGET_FR, N_TRIALS, has_rhythm, has_refractory, FREQ_SEQ, MU, kappa, FS, LEN_TRIAL)

    phase_singletrial, mus_single_trial, kappas_single_trial, true_len_trial = RePh.make_singletrial_phase()

    RePh.make_multitrial_phase()
    df = RePh.make_spike_probs()
    RePh.label_trials()

    RePh.df.to_csv(os.path.join(DATASAVEPATH,r'sim_RefractoryPhase_df.csv'))
    actual_frate = (df['spikes'].sum()/df.shape[0])*FS

    print('sim params: \n',
          'neuron id:' 'RePh',
          'avg_frate: ',actual_frate,'\n',
          'n_trials: ',N_TRIALS,'\n',
          'freq_seq: ',FREQ_SEQ,'\n',
          'mu: ',MU,'\n',
          'kappa: ',kappa,'\n',
          'fs:',FS,'\n',
          '\n',file=open(os.path.join(DATASAVEPATH,'ScAdv_sim_params_all/',"RePh_sim_params.txt"), "a"))
