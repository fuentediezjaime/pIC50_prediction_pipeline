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
import re
import warnings
# Import the custom configuration file
import config

# Import the custom packages 
from pIC50_predictor.features import load_smiles_from_csv, generate_fps
from pIC50_predictor.model import pIC50Predictor, find_best_params, OptimizerCallback


# Ignore a warning from lgbm when the features do not have names
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")


def get_next_run_num(results_dir: str) -> int:
    """
    Checks the existing results directory for the highest run number.
    """
    if not os.path.exists(results_dir):
        return 1 # Return a "falsy" value
    
    max_run_num = 0
    # Patrón para encontrar 'run_XXX' al principio de un nombre de fichero
    run_pattern = re.compile(r'run_(\d+)') #Compiling with the regex makes the matcher faster (not that it matters much here)
    
    for filename in os.listdir(results_dir):
        match = run_pattern.match(filename)
        if match:
            run_num = int(match.group(1))
            if run_num > max_run_num:
                max_run_num = run_num
                
    return max_run_num + 1




def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)



    next_run_number = get_next_run_num('results')
    fp_type_name = config.FP_TYPE
    
    # Formateamos el nombre con ceros a la izquierda para un ordenamiento correcto (ej. run_001, run_002)
    run_name = f"run_{next_run_number:03d}_{fp_type_name}"
    


    #Loading and preparing data
    print('Loading data...')
    train_smiles = load_smiles_from_csv(config.DATA_PATH, smiles_col=config.SMILES_COL)
    train_targets = pd.read_csv(config.TARGET_PATH)
    train_targets = train_targets[config.TARGET_COL]

    features = generate_fps(
        train_smiles,
        fp_type = config.FP_TYPE,
        **config.FP_PARAMS
    )
    print('Loaded data and parameters')

    opt_callback = OptimizerCallback(config.SEARCH_SPACE) #The optimization callback object that stores the search history


    #Here the hyperparameter search runs.
    best_params = find_best_params(
        X_train=features,
        y_train=train_targets,
        search_space=config.SEARCH_SPACE,
        n_calls=config.N_CALLS_OPTIMIZATION,
        cv_folds=config.CV_FOLDS,
        callbacker=opt_callback
    )


    #Saving training history.
    print('\n Saving search history')
    results_df=pd.DataFrame(opt_callback.results)
    results_df=results_df.sort_values(by='score').reset_index(drop=True)
    log_path = f'results/{run_name}_search_logs.csv'
    results_df.to_csv(log_path)
    print(f'Search history was saved in {log_path}')

    #Save a convergence plot
    f = plt.figure(figsize=(10,6))
    ax = f.add_axes((0.1,0.1,0.8,0.8))
    ax.plot(scores, marker='.', linestyle='--')
    ax.set_xlabel('Iteration number')
    ax.set_ylabel('Best MAE in CV')
    ax.set_title(f'Convergence: {run_name}')
    plot_path = f'results/{run_name}_convergence_plot.png'
    f.savefig(plot_path)
    print(f'Conv. plot saved in {plot_path}')

    #Now, we retrain with the optimum parameters on all the dataset
    final_predictor = PIC50Predictor(model_params=best_params)
    final_predictor.train(features, train_targets)
    print('I AM DONE')

if __name__ == '__main__':
    main()

