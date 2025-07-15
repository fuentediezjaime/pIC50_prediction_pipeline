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
import time

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


class pIC50Predictor:
    """
    Encapsulation of the LGBM tree model that does the prediction of pIC50 values.
    It implements its own predict, train load methods with safeguards to guarantee pretraining
    """

    def __init__(self, model_params: Dict[str, Any]=None):
        """
        Class constructor. initialize model with the parameters in model_params. Default is no parameters
        """
        self.model = lgb.LGBMRegressor(**(model_params or {}), random_state=33, verbose=-1) #We pass either the introduced model_params or an empty dict (so the default LGBM params)
        self._is_trained = False

    def train(self, X_train: List, y_train: np.ndarray) -> None: #No output, just change internal state
        self.model.fit(X_train, y_train)
        self._is_trained = True

    def predict(self, X_new: List) -> np.ndarray: #Returns "y_pred", the predicted pIC50 vals.
        if self._is_trained:
            return self.model.predict(X_new)
        else:
            raise RuntimeError('the model was not trained before predicting')
    def save(self, path: str) -> None:
        'Saves the trained predictor on a file'
        if self._is_trained:
            joblib.dump(self, path)
            print(f'The model was saved at {path}')

        else:
            raise RuntimeError('The model was not trained prior to saving')
        
    @classmethod #To avoid creating a dummy instance before loading the model
    def load(cls, path: str) -> 'pIC50Predictor':
        """Loads the predictor from a file and returns an instance of the predictor class"""
        predictor = joblib.load(path)
        print(f'Loaded the predictor from {path}')
        return predictor


def find_best_params(
        X_train: List,
        y_train: np.ndarray,
        search_space: List, #It is a list of the internal Integer, Real skopt objects, that have inner bounds.
        n_calls: int = 50,
        cv_folds: int = 10,
        callbacker: OptimizerCallback = None) -> Dict[str, Any]:
    """Function to launch the GP search of the best hyperparameters within a range."""
    np.int = int #skopt quirk.

    @use_named_args(search_space)
    def objective(**params):
        model = lgb.LGBMRegressor(random_state=33, verbose=-1, **params)
        score = -np.mean(cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='neg_mean_absolute_error'))
        return score

    print(f'\n Starting bayesian search. Total iterations: {n_calls}')

    #Now the core of the work:
    search_result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=n_calls,
        random_state=33,
        n_jobs=-1,
        callback=[callbacker] if callbacker else [] # Here the callback class that we created enters to log the history
    )

    best_params = {param.name : val for param, val in zip(search_space, search_result.x)}
    print(f"\nSearch is finished. best : {search_result.fun}")

    return best_params
