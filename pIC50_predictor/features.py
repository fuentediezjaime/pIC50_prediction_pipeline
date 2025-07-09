#pIC50_predictors/features.py

"""
Module to load the data and generate molecular descriptors.

The file contains functions for:
1) Load SMILES from the CSV dataset
2) Convert the SMILES strings in different kinds of molecular fingerprints.
3) Compare similarity between fingerprints to analyze the input structure.
"""

import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from typing import List, Union

def load_smiles_from_csv(path : str, smiles_col: Union[str, int] = 1) -> np.ndarray:
    """
    Loads the smiles from a CSV

    args:
        path (str): path to the CSV file containing smiles.
        smiles_col (Union[str, int]=1): the column of the CSV file containing the SMILES strings. It can be given as the DF's column name 
        or as the column index. Set to 1 by default.
    """

    df = pd.read_csv(path)
    if isinstance(smiles_col, int): # If smiles_col is a column index
        return df.iloc[:, smiles_col].values
    else: #If what is given is a column name
        return df[smiles_col].values
    


def generate_rdkit_fingerprints(smiles_list: List[str]) -> List[ExplicitBitVect]:
    """
    Generates standard rdkit fingerpin
    """