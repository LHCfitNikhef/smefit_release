# -*- coding: utf-8 -*-
"""Implement the corrections table basis roation"""
import numpy as np
import pandas as pd

from .compute_theory import flatten


def rotate_to_fit_basis(lin_dict, quad_dict, rotation_matrix):
    """
    Rotate to fitting basis

    Parameters
    ----------
        lin_dict: dict
            theory dictionary with linear operator corrections in the table
            basis
        quad_dict: dict
            theory dictionary with quadratic operator corrections in the table
            basis, emptry if quadratic corrections are not used
        rotation_matrix: pandas.DataFrame
            rotation matrix from tables basis to fitting basis

    Returns
        lin_dict_fit_basis: dict
            theory dictionary with linear operator corrections in the fit
            basis
        quad_dict_fit_basis: dict
            theory dictionary with quadratic operator corrections in the fit
            basis, emptry if quadratic corrections are not used
    """
    lin_df = pd.DataFrame(lin_dict)
    quad_dict_fit_basis = {}

    # select the columns corresponding to the
    # relevent corrections for the operator card basis
    R = rotation_matrix.loc[lin_df.columns, :]
    lin_dict_fit_basis = lin_df @ R

    # look at the quadratic corrections?
    if quad_dict == {}:
        return lin_dict_fit_basis.to_dict("list"), quad_dict_fit_basis

    tensor = []
    # loop over table basis pairs
    # and build an (n_op_table, n_dat, n_op_fit, n_op_fit) tensor
    for col, values in quad_dict.items():
        o1, o2 = col.split("*")
        r1 = R.loc[o1, :]
        r2 = R.loc[o2, :]
        r1r2 = np.outer(r1, r2)
        r1r2o1o2 = np.einsum("i,kj->ikj", values, r1r2)
        tensor.append(r1r2o1o2)

    # sum over table basis entries
    tensor = np.array(tensor)
    new_quad_corrections = tensor.sum(axis=0)

    # flatten the tensor (n_dat, n_op_fit, n_op_fit) -> (n_dat, n_op_fit_pairs)
    new_quad_matrix = flatten(new_quad_corrections, index=1)

    # build the new quadratic corrections dictionary
    for i, r1 in enumerate(rotation_matrix.columns):
        for r2 in rotation_matrix.columns:
            new_key = f"{r1}*{r2}"
            if new_key not in quad_dict_fit_basis:
                quad_dict_fit_basis[new_key] = new_quad_matrix[:, i]

    return lin_dict_fit_basis.to_dict("list"), quad_dict_fit_basis
