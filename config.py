"""
Conf_file for the parameter search.
"""
from skopt.space import Real, Integer

# FOR THE HYPERPARAMETER SEARCH

# Definition of the search space.
SEARCH_SPACE = [
    Integer(15, 50, name='max_depth'),
    Real(1e-3, 5e-2, "log-uniform", name='learning_rate'),
    Integer(1500, 4000, name='n_estimators'),
    Real(1e-2, 10.0, "log-uniform", name='reg_lambda'),
    Real(0.05, 0.3, name='feature_fraction'),
    Integer(300, 1024, name='num_leaves')
]









# Define iterations of the bayes search, and the folds for each point in the search.
N_CALLS_OPTIMIZATION = 50  # Número de iteraciones para la búsqueda Bayesiana.
CV_FOLDS = 10              # Número de folds para la validación cruzada.



#PATHS, COLUMN NAMES...

# Paths to data files.
DATA_PATH = "dataset/X_train_AX2CWD7.csv"
TARGET_PATH = "dataset/y_train_hCIvDMj.csv"

# Column names.
SMILES_COL = 'smiles'
TARGET_COL = 'y'

# Fingerprint configurations.
FP_TYPE = 'morgan'
FP_PARAMS = {'radius': 2, 'nBits': 2048}
