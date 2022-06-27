# Spike-Phase & History Models

Here you will find a series of scripts and functions to fit spike-phase models and spike histories. 

Phase models rely on fitting kernel density estimators to the distribution of rhythmic phase values that coincide with a neuron's spikes. The kernel bandwidth is optimized via cross-validation. 

History models are fit via logistic regression. 

The scripts to run the models all begin with the "run_models_" prefix, and they vary according to the format the data are in (.mat files versus .csv files) as well as whether there is an option for leave-one-out cross-validation to find the optimal KDE kernel bandwidth. For instance, "run_all_models_simcells.py" will expect data to be already organized as a dataframe, while "run_all_models_realcells.py" will expect data to come in as a .mat file. 
