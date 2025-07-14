'''
This is the main launcher script to run my hyperparam search. It does the following:

1) Loads the configuration parameters (folds for crossval and ranges for hyperparameters) and the data.
2) Optimizes the hyperparameters via bayes search
3) Saves a log and a plot of the optimization
4) Uses the optimum hyperparams to retrain a model on all the data
5) Saves the trained optimum model
'''


import pandas as pd
import matplotlib.pyplot as plt
import os


# Import the custom configuration file
import config

# Import the custom packages 
from pic50_predictor.features import load_smiles_from_csv, generate_fingerprints
from pic50_predictor.model import PIC50Predictor, find_best_hyperparameters, OptimizerCallback

