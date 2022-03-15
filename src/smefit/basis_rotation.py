# -*- coding: utf-8 -*-
"""Implement the corrections table basis roation"""
import numpy as np
import pandas as pd

from .compute_theory import flatten


def rotate_to_fit_basis(theory_dict, rotation_matrix, use_quad):
    """
    Rotate to fitting basis

    Parameters
    ----------
        theory_dict: dict
            theory dictionary with operator corrections in the table
            basis
        rotation_matrix: pandas.DataFrame
            rotation matrix from tables basis to fitting basis

    Returns
        new_theory_dict:
            theory dictionary with operator corrections in the table
            basis

    """
    quad_dict = {}
    lin_dict = {}
    for key, value in theory_dict.items():

        if "*" in key and use_quad:
            quad_dict[key] = value
        elif "^" in key:
            quad_dict[f"{key[:-2]}*{key[:-2]}"] = value
        elif key == "SM":
            continue
        lin_dict[key] = value

    lin_df = pd.DataFrame(lin_dict)
    # select the columns corresponding to the
    # relevent corrections for the operator card basis
    R = rotation_matrix.loc[lin_df.colums, :]
    new_lin_dict = lin_df @ R

    tensor = []
    # look at the quadratic corrections
    for col, values in quad_dict.items():
        o1, o2 = col.split("*")
        r1 = R.loc[o1, :]
        r2 = R.loc[o2, :]
        r1r2 = np.outer(r1, r2)
        r1r2o1o2 = np.einsum("i,kj->ikj", values, r1r2)
        tensor.append(r1r2o1o2)

    tensor = np.array(tensor)
    new_quad_corrections = tensor.sum(axis=0)

    new_quad_matrix = flatten(new_quad_corrections, index=1)
    new_quad_dict = {}

    for i, r1 in enumerate(rotation_matrix.columns):
        for r2 in rotation_matrix.columns:
            new_key = f"{r1}*{r2}"
            if new_key not in new_quad_dict:
                new_quad_dict[new_key] = new_quad_matrix[:, i]

    return new_lin_dict.to_dict("list"), new_quad_dict
