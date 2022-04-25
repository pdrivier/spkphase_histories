# Computes the rayleigh mean resultant length, resultant length, z statistic, p-value, and
# circular mean of the phase angles that coincide with spikes

import numpy as np
import pandas as pd

def rayleigh_pr(df,rhythm_low_high):

    """Code is based on Phillip Berens code on github: https://github.com/philippberens/PyCircStat
    df: dataframe, must contain 'spikes' column, and phase column with names that
        follow [rhythm]_[low]_[high]_phases naming convention
    rhythm_low_high:  string, e.g. 'theta_4_12'"""

    spikes = df['spikes'].values

    rhythmic_phases = rhythm_low_high + '_phases'
    phases = df[rhythmic_phases].values

    #find all phases that co-occur with spike event
    phi_when_sp = []
    for s,p in zip(spikes,phases):

        if s==1:
            phi_when_sp.append(p)

    #get cosine of all phase angles that coincide with spikes
    tmpcos = [np.cos(i) for i in phi_when_sp]
    #get sine of all phase angles that coincide with spikes
    tmpsin = [np.sin(i) for i in phi_when_sp]

    #take the average cosine component (x)
    meancos = np.mean(tmpcos)
    #take the average sine component (y)
    meansin = np.mean(tmpsin)

    #get the hypotenuse of the mean sin and cosine components
    mrl = np.sqrt(meancos**2 + meansin**2)

    #compute circular mean
    circmean = np.arctan2(meansin,meancos)

    #grab number of observations of spike-phase coincidences
    n = len(phi_when_sp)

    #resultant length (removes normalization inherent in mrl, need it for pval computation)
    R = n*mrl

    ray_d = {'rhythm': [rhythm_low_high],
             'mrl': [mrl],
             'R': [R],
             'z': [R**2 / len(phi_when_sp)],
             'circmean': [circmean],
             'pval': [np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))]
            }

    ray_df = pd.DataFrame(ray_d)

    return ray_df
