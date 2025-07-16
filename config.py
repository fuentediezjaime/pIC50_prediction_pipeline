"""
Conf_file for the parameter search.
"""
from skopt.space import Real, Integer

# FOR THE HYPERPARAMETER SEARCH

# Definition of the search space.
SEARCH_SPACE = [
    Integer(5, 50, name='max_depth'),
    Real(1e-3, 5e-2, "log-uniform", name='learning_rate'),
    Integer(500, 5000, name='n_estimators'),
    Real(0.01, 10.0, "log-uniform", name='reg_lambda'),
    Real(0.01, 0.5, name='feature_fraction'),
    Integer(300, 1024, name='num_leaves')
]









# Define iterations of the bayes search, and the folds for each point in the search.
N_CALLS_OPTIMIZATION = 10 #50  # It. number for bayesian search
CV_FOLDS = 2 #5 or 10              # Folds number for crossval at each bayes search step.




# Paths to data files.
DATA_PATH = "dataset/X_train_AX2CWD7.csv"
TARGET_PATH = "dataset/y_train_hCIvDMj.csv"

# Column names.
SMILES_COL = 'smiles'
TARGET_COL = 'y'

# Fingerprint configurations.
FP_TYPE = 'mor_rdk_scalars'#'mor_rdk'
FP_PARAMS = {'radius': 2, 'nBits': 1024, 'maxPath': 7, 'fpSize': 1024}
