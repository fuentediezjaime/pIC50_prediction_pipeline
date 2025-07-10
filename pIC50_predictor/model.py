"""
This module contains the class pIC50 predictor, which contains a LightGBM model and functions for hyperparameter search.
"""


import lightgbm as lgb
import joblib
import numpy as np
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space import Real, Integer
from typing import List, Dict, Any
from skopt.utils import use_named_args
from typing import List, Dict, Any, Tuple

class OptimizerCallback:
    """
    This is the class that registers the results of gp_minimize at each iteration.
    """
    def __init__(self, search_space: List):
        self.search_space = search_space
        self.results = []

    def __call__(self, res): #Allows to call the class directly. res is the output of the gp_minimize optimizer
        latest_params = {param.name: val for param, val in zip(self.search_space, res.x_iters[-1])}
        latest_score = res.func_vals[-1]
        self.results.append({**latest_params, 'score': latest_score})
        print(f"Search iteraton {len(self.results)}: score={latest_score:.4f}")


class pIC50Predictor