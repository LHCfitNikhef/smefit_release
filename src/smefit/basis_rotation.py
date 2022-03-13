# -*- coding: utf-8 -*-
"""Implement the corrections table basis roation"""
import pandas as pd


def rotate_to_fit_basis(theory_dict, rotation_matrix):
    """
    Rotate to fitting basis

    Parameters
    ----------
        theory_dict: dict
            theory dictionary with operator corrections in the table
            basis
        rotation_matrix: numpy.ndarray
            rotation matrix from tables basis to fitting basis

    Returns
        new_theory_dict:
            theory dictionary with operator corrections in the table
            basis

    """

    pd.DataFrame(theory_dict)

    return theory_dict
