#Kolmogorov Smirnov plots and time rescaling

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem


def kl(probs1,probs2):
    if len(probs1) != len(probs2):
        raise ValueError('arguments must be same size.')
    kls = []
    for (p1,p2) in zip(probs1,probs2):
        kls.append(p1*np.log2(p1/p2)+(1-p1)*np.log2((1-p1)/(1-p2)))
    return kls


def compute_bic(logloss,n,k):
    """Compute the BIC for n observations and some logloss model value, of order
        k"""

    bic = k*np.log2(n) - 2*np.log2(logloss)

    return bic


def rescale(spikes,probs):
    '''Rescale observed interspike intervals according to model conditional intensity'''
    # get the times of spikes
    spiketimes = np.nonzero(spikes)[0]
    # the CDF of rescaled spikes
    rescaled_cdf = []
    for i in range(len(spiketimes)-1):
        # get the indices of
        spike = spiketimes[i]
        next_spike = spiketimes[i+1]
        # get the new interspike arrival time under rescaling
        rescaled_isi = sum(probs[spike:next_spike])
        rescaled_cdf.append(1 - np.exp(-rescaled_isi))
    return sorted(rescaled_cdf)

def rescale_discrete(spikes,probs):
    """Apply analytical correction to rescaled ISIs for the discrete time case"""

    #get the times of spikes
    spiketimes = np.nonzero(spikes)[0]

    #make the qk's (Haslinger, Pipa, & Brown, 2010)
    qk = -np.log(1-probs)

    #the CDF of rescaled spikes
    rescaled_cdf = []
    for i in range(len(spiketimes)-1):

        #initialize random numbers
        r = np.random.random_sample()

        total=[] #reset

        spike = spiketimes[i]
        next_spike = spiketimes[i+1]

        total = sum(qk[spike+1:next_spike-1])
        delta = -(1/qk[next_spike])*np.log(1-r*(1-np.exp(-qk[next_spike])))

        total = total+qk[next_spike]*delta
        risi = 1 - np.exp(-total)
        rescaled_cdf.append(risi)

    rescaled_cdf = sorted(rescaled_cdf)

#     inrst = 1/(len(spikes)-1)
#     xrs = 0.5*inrst:inrst:1-0.5*inrst
    return rescaled_cdf


def ksPlot(cdf,ax=None,color=None):
    """Create KS plot from rescaled conditional density function"""

    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=(15,15))

    N = len(cdf)
    ax.plot(cdf,(np.arange(N)+0.5)/N,color)
    ax.plot([0,1],[0,1],'k',label='_nolegend_')
    ax.plot([0,1],[1.36/np.sqrt(N),1+1.36/np.sqrt(N)],'k--',label='_nolegend_')
    ax.plot([0,1],[-1.36/np.sqrt(N),1-1.36/np.sqrt(N)],'k--',label='_nolegend_')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    return ax


def logloss(probs,y):
    """Computes the average log loss across all timepoints to evaluate a model's predictions"""
    avg_loss = 0
    all_loss = []
    if probs.size > 1:
        for i,observation in enumerate(y):
            if observation == 1:
                avg_loss += -np.log2(probs[i])
                all_loss.append(-np.log2(probs[i]))
            else:
                avg_loss += -np.log2(1 - probs[i])
                all_loss.append(-np.log2(1 - probs[i]))
        avg_loss = avg_loss/len(y)
        sem_loss = sem(all_loss)
    else:
        for i,observation in enumerate(y):
            if observation == 1:
                avg_loss += -np.log2(probs)
            else:
                avg_loss += -np.log2(1 - probs)
        avg_loss = avg_loss/len(y)
    return avg_loss, sem_loss, all_loss

def logistic(log_odds):
    """Compute logistic probabilities from the log odds"""

    probs = np.exp(log_odds)/(1 + np.exp(log_odds))
    return probs


def logodds(probs):
    """Compute the log odds"""

    # use natural logarithm
    mu = np.log(probs/(1-probs))
    return mu

def bayes(blikelihood,aprior,bprior):
    """Computes posterior conditional probability using Bayes Theorem"""

    aposterior = (blikelihood*aprior)/bprior

    return aposterior
