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
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

def load_smiles_from_csv(path : str, smiles_col: Union[str, int] = 1) -> np.ndarray:
    """
    Loads the smiles from a CSV

    args:
        path (str): path to the CSV file containing smiles.
        smiles_col (Union[str, int]=1): the column of the CSV file containing the SMILES strings. It can be given as the DF's column name 
        or as the column index. Set to 1 by default.

    Return:
        np.ndarray: a numpy array filled with SMILES strings
    """

    df = pd.read_csv(path)
    if isinstance(smiles_col, int): # If smiles_col is a column index
        return df.iloc[:, smiles_col].values
    else: #If what is given is a column name
        return df[smiles_col].values
    


def generate_fps(smiles_list: List[str], fp_type: str, **kwargs) -> List[ExplicitBitVect]:
    """
    Generates a specific type of molecular fingerprint from RDKIT.
    Args:
        smiles_list (List[str]): List of smiles strings.
        fp_type (str): keyword indicating the fingerprint to be generated ('rdkit', 'morgan', 'maccs')
        **kwargs: keywords for the selected fingerprint type


    """

    fp_type = fp_type.lower() #Ensure that the keyword comes in lowercase
    fingerprints = []

    for smile in smiles_list: #Generate the mol objects
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            continue #Skip smiles that failed to convert.
        if fp_type =='rdkit':
            fp = Chem.RDKFingerprint(mol, **kwargs)
        elif fp_type == 'morgan':
            radius = kwargs.get('radius',2)
            nBits = kwargs.get('nBits',2048)
            fpgen = GetMorganGenerator(radius, fpSize=nBits)
            fp = fpgen.GetFingerprint(mol)
        elif fp_type == 'maccs':
            fp = MACCSkeys.GenMACCSKeys(mol)
        else:
            raise ValueError(f'Not supported fingerprint type: {fp_type}')            
        fingerprints.append(fp)

    return fingerprints


def tanimoto_similarity(fps: List[ExplicitBitVect]) -> np.ndarray:
    """
    The function computes tanimoto similarity between all the fingerprints in the provided list. Returns a "similarity matrix".
    Args:
        fps (List[ExplicitBitVect]): List with the fingerprints computed from rdkit

    Returns:
        np.ndarray: a matrix with the similarities.
    """

    n_fps = len(fps)
    sim_mat = np.zeros((n_fps,n_fps))
    for i in range(num_fps):
        for j in range(i, num_fps):
            sim = DataStructs.FingerprintSimilarity(fps[i], fps[j])
            sim_mat[i, j] = sim
            sim_mat [j, i] = sim

    return sim_mat # sim_mat will have its diag populated by zeros.
