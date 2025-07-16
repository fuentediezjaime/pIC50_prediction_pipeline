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
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, Lipinski
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
            maxpath= kwargs.get('maxPath',7)
            fpsize = kwargs.get('fpSize', 2048)
            fpgen_rdk = AllChem.GetRDKitFPGenerator(maxPath=maxpath, fpSize=fpsize)
            fp = fpgen_rdk.GetFingerprint(mol)


        elif fp_type == 'morgan':
            radius = kwargs.get('radius',2)
            nBits = kwargs.get('nBits',2048)
            fpgen = GetMorganGenerator(radius, fpSize=nBits)
            fp = fpgen.GetFingerprint(mol)


        elif fp_type == 'maccs':
            fp = MACCSkeys.GenMACCSKeys(mol)


        elif fp_type =='mor_rdk':
            # The morgan fp is created.
            radius = kwargs.get('radius',2)
            nBits = kwargs.get('nBits',2048)
            fpgen = GetMorganGenerator(radius, fpSize=nBits)
            fp_mor = fpgen.GetFingerprint(mol)

            #The RDKIT (topological) fingerprint is created.
            maxpath= kwargs.get('maxPath',7)
            fpsize = kwargs.get('fpSize', 1024)
            fpgen_rdk = AllChem.GetRDKitFPGenerator(maxPath=maxpath, fpSize=fpsize)
            fp_rdk = fpgen_rdk.GetFingerprint(mol)

            # Fuse the fingerprints
            fp = fp_mor + fp_rdk


        elif fp_type == 'mor_rdk_scalars': #This fingerprint concatenates the morgan fingerprints and some scalar values from RDKit.
            molwt_raw = Descriptors.MolWt(mol)
            log_p_raw = Descriptors.MolLogP(mol)
            n_Hdon_raw = Lipinski.NumHDonors(mol)
            n_Hacc_raw = Lipinski.NumHAcceptors(mol)
            tpsa_raw = Descriptors.TPSA(mol)

            #Some smiles strings result in some of this quantities being tuples sometimes, sometimes floats. Check type and convert.
            molwt = molwt_raw[0] if isinstance(molwt_raw, tuple) else molwt_raw
            log_p = log_p_raw[0] if isinstance(log_p_raw, tuple) else log_p_raw
            n_Hdon = n_Hdon_raw[0] if isinstance(n_Hdon_raw, tuple) else n_Hdon_raw
            n_Hacc = n_Hacc_raw[0] if isinstance(n_Hacc_raw, tuple) else n_Hacc_raw
            tpsa = tpsa_raw[0] if isinstance(tpsa_raw, tuple) else tpsa_raw[0]  

            scalars_array = np.array([molwt, log_p, n_Hdon, n_Hacc, tpsa])
            radius = kwargs.get('radius',2)
            nBits = kwargs.get('nBits',2048)
            fpgen = GetMorganGenerator(radius, fpSize=nBits)
            fp_mor = np.array(fpgen.GetFingerprint(mol))

            #The RDKIT (topological) fingerprint is created.
            maxpath= kwargs.get('maxPath',7)
            fpsize = kwargs.get('fpSize', 1024)
            fpgen_rdk = AllChem.GetRDKitFPGenerator(maxPath=maxpath, fpSize=fpsize)
            fp_rdk = np.array(fpgen_rdk.GetFingerprint(mol))


            fp = np.concatenate([fp_mor, fp_rdk, scalars_array])




        elif fp_type == 'mor_rdk_scalar':
            pass

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
