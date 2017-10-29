# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pandas as pd

from datawarehouse.higgsml import tau_energy_scale
from datawarehouse.higgsml import label_to_float


__doc__ = "Helper function to preprocess and skew the data."
__version__ = "1.0"
__author__ = "Victor Estrade"


def handle_missing_values(data, missing_value=-999.0, dummy_variables=False):
    """
    Find missing values.
    Replace missing value (-999.9) with 0.
    If dummy_variables is created then :
        for each feature having missing values, add a 'feature_is_missing' boolean feature.
        'feature_is_missing' = 1 if the 'feature' is missing, 0 otherwise.
    
    Args
    ----
        data : (pandas.DataFrame) the data with missing values.
        missing_value : (default=-999.0) the value that should be considered missing.
        dummy_variables : (bool, default=False), if True will add boolean feature columns 
            indicating that values are missing.
    Returns
    -------
        filled_data : (pandas.DataFrame) the data with handled missing values.
    """
    is_missing = (data == missing_value)
    filled_data = data[~is_missing].fillna(0.)  # Magik
    if dummy_variables :
        missing_col = [c for c in is_missing.columns if np.any(is_missing[c])]
        new_cols = {c: c+"_is_missing" for c in missing_col}  # new columns names
        bool_df = is_missing[missing_col]  # new boolean columns
        bool_df = bool_df.rename(columns=new_cols)
        filled_data = filled_data.join(bool_df)  # append the new boolean columns to the data
    return filled_data


def clean_columns(data):
    """
    Removes : EventId, KaggleSet, KaggleWeight
    Cast labels to float.
    """
    data = data.drop(["EventId", "KaggleSet", "KaggleWeight",], axis=1)
    label_to_float(data)  # Works inplace
    return data


def skew(data, sysTauEnergyScale=1.0, cut_22GeV=True, missing_value=0., remove_mass_MMC=True):
    """
    """
    data_skewed = data.copy()
    if not "DER_mass_MMC" in data_skewed.columns:
        data_skewed["DER_mass_MMC"] =  np.zeros(data.shape[0]) # Add dummy column

    tau_energy_scale(data_skewed, sysTauEnergyScale, missing_value=missing_value)  # Modify data inplace
    data_skewed = data_skewed.drop(["ORIG_mass_MMC", "ORIG_sum_pt"], axis=1)

    if cut_22GeV:
        data_skewed = data_skewed[data_skewed['PRI_tau_pt'] > 22.0]
    if remove_mass_MMC and "DER_mass_MMC" in data_skewed.columns:
        data_skewed = data_skewed.drop( ["DER_mass_MMC"], axis=1 )
    return data_skewed


def log_scale(data, inplace=True):
    cols = ["DER_mass_MMC",
            "DER_mass_transverse_met_lep",
            "DER_mass_vis",
            "DER_pt_h",
            "DER_mass_jet_jet",
            "DER_pt_tot",
            "DER_sum_pt",
            "DER_pt_ratio_lep_tau",
            "PRI_tau_pt",
            "PRI_lep_pt",
            "PRI_met",
            "PRI_met_sumet",
            "PRI_jet_leading_pt",
            "PRI_jet_subleading_pt",
            "PRI_jet_all_pt",
            ]
    if not inplace:
        data = data.copy()
    for c in cols :
        if c in data.columns:
            data[c] = np.log( 1 + data[c] )
    return data

