# Spike-Phase & History Models

Here you will find a series of scripts and functions to fit spike-phase models and spike histories. 

Phase models rely on fitting kernel density estimators to the distribution of rhythmic phase values that coincide with a neuron's spikes. The kernel bandwidth is optimized via cross-validation. 

History models are fit via logistic regression. 

The scripts to run the models all begin with the "run_models_" prefix, and they vary according to the format the data are in (.mat files versus .csv files) as well as whether cross-validation proceeds via a leave-one-out train-test protocol, or not.  
