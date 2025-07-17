# Pipeline for prediction pIC50 from SMILES strings.

This project was written originally to compete in the 2023 edition of the ENS Data Challenge "Predicting molecule-protein interaction for drug discovery". Although it is light enough to run on a laptop, it has been built with a more sophisticated production environment in mind. 

Conda (from MiniConda) was used as the virtual environment manager for the code due to the RDKIT compatibility. However, RDKIT can be compiled from source and installed on a venv environment.


## How to run.
The process of fingerprint generation and parameter search is automated. One just needs to ensure that the dataset and the target path are correctly indicated in the config.py file.

The parameters for the hyperparameter search are also defined in config.py, notably the ranges for the hyperparameters, the number of bayesian search iterations, and the number of cross-validation folds for each bayesian search step. If the search takes too long, the cross-validation fold number can be reduced to control cost.

Once the config file is set and the environment created, just run:


```bash
python3 train.py
```
It will output the MAE loss at every Bayes-search step and will automatically provide an optimum model retrained in the whole dataset.

Multiple consecutive runs do not overwrite the results. Different indices are assigned to consecutive run results, so taht they do not overwrite. They are stored in a directory called results/.

## Installation

1.  **Clone this repo:**
    ```bash
    git clone [https://github.com/fuentediezjaime/pIC50_prediction_pipeline.git](https://github.com/fuentediezjaime/pIC50_prediction_pipeline.git)

    cd pIC50_prediction_pipeline
    ```

2.  **Create and activate the Conda environment:**
    `environment.yml` contains all the dependencies.
    ```bash
    conda env create -f environment.yml
    conda activate pic50
    ```

3.  **Install the project in editable mode:**
    ```bash
    pip install -e .
    ```

# Results
## Simple fingerprints: Morgan - RDKIT - MACCS
To start we compare the results using 3 popular types of fingerprints directly available on RDKIT, Morgan fingerprints, RDKIT fingerprints and MACCS keys. All 3 are binary bit-vectors. For all of them, the following hyperparameter space has been explored:

```
SEARCH_SPACE = [
    Integer(15, 50, name='max_depth'),
    Real(1e-3, 5e-2, "log-uniform", name='learning_rate'),
    Integer(1500, 4000, name='n_estimators'),
    Real(1e-2, 10.0, "log-uniform", name='reg_lambda'),
    Real(0.05, 0.3, name='feature_fraction'),
    Integer(300, 1024, name='num_leaves')
]
```

The fact that most of the optimum hyperparameters were inside the bounds of the search range indicates that the model is not "pushing the boundaries" of the allowed hyperparameter space. Below are the MAEs of the different fingerprints.

|Fingerprint|MAE|
|---|---|
|Morgan(2)|0.534|
|Morgan(3)|0.549|
|RDKIT|0.563|
|MACCS| 0.631|
|Morgan + rdkit| 0.532|


We must improve the description of the molecules in order to further lower the MAE. We see that hyperparameter search alone is not enough, and neither are single RDKIT fingerprints. A common solution is to concatenate fingerprints, hopefully fingerprints that contain non-redundant information to improve the "expresiveness" of our fingerprints. An example of this are the topological fingerprints from RDKit, which work by identifying connection paths through the molecular structure, while the Morgan fingerprints describe the circular environment of the atoms. However, concatenating the FPs does not improve the MAE of the model.

## Adding physico-chemical descriptors.
It seems that the topological description of the molecule is not enough to get a great predictive performance from the model. In principle, as tree-based learners do not suffer from differences in the scale of the input parameters, (1/0 bits vs scalars, for instance), we can add scalar physico-chemical descriptors from RDKIT.


## Pruning the unnecessary digits.
Until now, we've progressively increased the complexity of the fingerprint, and that has given us quite modest jumps in performance. Also, when one performs hyperparameter search, the optimum models show a maximal number of trees, and a close to minimal learning rate. This means that the model is becoming "squeezed". Large increases in capacity are needed to explore the minor details of the error function and that, together with very small learning rates, allow our models to find very shallow minima in the hyperparameter space. 

Still, one expects that, if a fingerprint becomes "twice as informative" (by concatenating two fingerprints"), the error should sharply fall. A possible explanation of this contradiction is that most bits on the RDKIT or the Morgan fingerprint are redundant or non informative, and the information is really conveyed by a small subset of the features. Luckily, from the lightGBM models an importance ranking for each feature can be extracted.

To ensure consistency, we run 3 hyperparameter searches with identical search ranges, on morgan+rdkit+scalar fingeprints. We compute the average importance of the features along this 3 runs
