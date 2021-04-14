#a class to package the nitty gritty code for Entropy2020


import itertools
import numpy as np
import pandas as pd
from process_trial_data import find_true_trial_end_samples

class Automaze_Spkphase:

    def __init__(self,data,session_id,pre_trial_time,fs,long,short):
        self.data = data
        self.session_id = session_id
        self.pre_trial_time = pre_trial_time #in sec
        self.fs = fs
        self.long = long  #in sec
        self.short = short #in sec

    def extra_msec(self):

        self.hist_longest = int(self.pre_trial_time*self.fs)

        return self.hist_longest


    def get_fulldata(self,spikes_var_name,phase_var_name,lfp_var_name):

        #grab the full dataset
        self.All_Spikes = self.data[spikes_var_name]
        self.All_Phases = self.data[phase_var_name]
        self.All_LFP = self.data[lfp_var_name]

        return self.All_Spikes, self.All_Phases, self.All_LFP

    def process_mat_struct(self):
        """Process .mat data files containing spike and phase data.

        Params:
        data:............sio.loadmat output
        pre_trial_time:.. time duration in msec that has been added to
        .................each trial to get history for the first trial times
        """

        #number of trials per session:
        self.n_trials = self.data['spikes_stacked'].shape[0]

        #size of each trial, including the extra time added pre trial start
        len_trials = self.data['spikes_stacked'].shape[1]

        #length of relevant trial epoch (in msec)
        true_len_trials = len_trials-self.extra_msec()

        #get length of dataset
        n_all_samples = len(self.All_Spikes)

        #get select samples (belonging to legit trial data)
        n_cut_samples = n_all_samples - self.extra_msec()*self.n_trials

        self.trial_data = {'n_trials': self.n_trials,
                              'len_trials': len_trials,
                              'true_len_trials':true_len_trials,
                              'n_all_samples': n_all_samples,
                              'n_cut_samples': n_cut_samples}

        #account for lack of continuity between trials
        self.trial_ends = [(len_trials*i) - 1 for i in np.arange(1,self.n_trials+1)]
        iter_timesamps_lists = [np.arange(i-(len_trials-1),(i-self.hist_longest)+1) for i in self.trial_ends]
        self.iter_timesamps = list(itertools.chain.from_iterable(iter_timesamps_lists))

        return self.trial_data, self.iter_timesamps, self.trial_ends, self.n_trials

    def get_history_times(self):

        return self.long, self.short

    def history_in_msec(self):

        self.longmsec = int(self.long*self.fs)
        self.shortmsec = int(self.short*self.fs)

        return self.longmsec, self.shortmsec

    def make_history(self):

        #Build master history dataframe
        Trial_History = np.zeros((self.extra_msec(),
                                  self.trial_data.get('n_cut_samples')))
        history=[]
        for ind,j in enumerate(self.iter_timesamps):

            history = self.All_Spikes[j:j+self.extra_msec()]
            Trial_History[:,ind] = history

        histdf = pd.DataFrame(data=Trial_History.T)
        histdf = histdf[histdf.columns[::-1]] #reverse column order

        Trial_Spikes=pd.Series([],name='spikes')
        Trial_Phases=pd.Series([],name='phase')
        Trial_LFP=pd.Series([],name='lfp')

        n = self.trial_data.get('true_len_trials')
        for i in self.trial_ends:
            Trial_Spikes = Trial_Spikes.append(pd.Series(data=self.All_Spikes[i-(n-1):i+1]),ignore_index=True)
            Trial_Phases = Trial_Phases.append(pd.Series(data=self.All_Phases[i-(n-1):i+1]),ignore_index=True)
            Trial_LFP = Trial_LFP.append(pd.Series(data=self.All_LFP[i-(n-1):i+1]),ignore_index=True)

        self.Trial_Spikes = Trial_Spikes.rename('spikes')
        self.Trial_Phases = Trial_Phases.rename('phase')
        self.Trial_LFP = Trial_LFP.rename('lfp')

        self.df = pd.concat([self.Trial_LFP, self.Trial_Phases, self.Trial_Spikes, histdf],axis=1)

        return self.df,self.Trial_Spikes,self.Trial_Phases, self.Trial_LFP


    def label_trials(self):
        # create column of trial labels, which will make trial-based cross validation
        # simpler
        true_trial_bins = find_true_trial_end_samples(self.trial_data.get('n_trials'),
                                                      self.trial_data.get('true_len_trials'))
        true_trial_bins.insert(0,0)
        labels = np.arange(len(true_trial_bins)-1)

        self.df['trial_labels'] = pd.cut(self.df.index.values,
                                         bins=true_trial_bins,
                                         labels=labels,
                                         include_lowest=True)

        return self.df

    def train_test_split_by_trials(self,test_size):
        """Partitions the dataset into train and test splits by trial label, that is,
        keeping all consecutive trial timestamps together"""

        n_trials_test = round(self.n_trials*test_size)
        n_trials_train = self.n_trials - n_trials_test

        all_labels = np.arange(self.n_trials)
        train_labels = random.choices(all_labels, k=n_trials_train)

        self.train = df[df['trial_labels'].isin(train_labels)]
        self.test = df[~df['trial_labels'].isin(train_labels)]

        return self.train, self.test

    def kde_optimal_bandwidth(self):
        train_spk_phase=[]
        for s,p in zip(self.train['spikes'].values,self.train['phase'].values):
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
        self.kde_bandwidth = grid.best_params_

        self.kde = grid.best_estimator_

        return self.kde, self.kde_bandwidth

    def compute_kde_estimator(self,phases_for_estimation):

        pdf = np.exp(selfkde.score_samples(phases_for_estimation[:, None]))
        self.pdf = pdf * 3 #normalize for the extra data added on either side in kde_optimal_bandwidth

        return self.pdf

    def compute_posterior(self,phase_grid=None,eval_train=None,eval_test=None):
        #how to select, on the spot, whether will do this for phase_grid, train, or test phases?

        phase_prior = 1/(2*np.pi)
        if not phase_grid==None:

            spikes_prior_test = self.test['spikes'].sum()/self.test.shape[0]
            self.pdf_post_singleCycle_test = bayes(self.pdf,
                                                   spikes_prior_test,
                                                   phase_prior)


        if not eval_train==None:

            spikes_prior_train = self.train['spikes'].sum()/self.train.shape[0]
            self.pdf_post_allTimes_train = bayes(self.pdf,
                                                 spikes_prior_train,
                                                 phase_prior)

        if not eval_test==None:

            spikes_prior_test = self.test['spikes'].sum()/self.test.shape[0]
            self.pdf_post_allTimes_test = bayes(self.pdf,
                                                spikes_prior_test,
                                                phase_prior)

        return self.pdf_post_singleCycle_test, self.pdf_post_allTimes_train, self.pdf_post_allTimes_test

    # def compute_log_odds(self):


    # def build_logReg_regressor(self):

    def label_phase_cycles(self,df_col_name_phase,df_col_name_cyclelabels):
        """df_col_name, string, e.g. 'phase"""

        self.df['phase_max2pi'] = self.df[df_col_name_phase] + np.pi

        st=self.df['phase_max2pi'].iloc[0]

        true_inds_within_cycle = np.where((self.df['phase_max2pi'] <= st))
        large_jumps = np.where(np.diff(true_inds_within_cycle[0])>1)

        cycle_edges = true_inds_within_cycle[0][large_jumps]
        cycle_edges = np.hstack((0,cycle_edges,self.df.index[-1]))

        labels = np.linspace(0,len(cycle_edges),len(cycle_edges)-1)

        self.df[df_col_name_cyclelabels] = pd.cut(self.df.index,
                                             bins=cycle_edges,
                                             labels=labels,
                                             include_lowest=True
                                            )
        self.df[df_col_name_cyclelabels] = self.df[df_col_name_cyclelabels].astype(int)

        return self.df
