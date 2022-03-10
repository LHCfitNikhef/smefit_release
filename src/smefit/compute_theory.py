# -*- coding: utf-8 -*-

"""
Module for the generation of theory predictions
"""
import numpy as np


def flatten(quad_mat):
    """
    Delete lower triangular part of a quadratic matrix
    and flatten it into an array
    """
    size = quad_mat.shape[0]
    return quad_mat[np.triu_indices(size)]


def make_predictions(dataset, coefficients_values, use_quad):
    """
    Generate the corrected theory predictions for dataset
    given a set of |SMEFT| coefficients.

    Parameters
    ----------
        dataset : DataTuple
            dataset tuple
        coefficients_values : numpy.ndarray
            |EFT| coefficients values
        use_quad: bool
            if True include also |HO| corrections
    Returns
    -------
        corrected_theory : numpy.ndarray
            SM + EFT theory predictions
    """

    # Compute total linear correction
    summed_corrections = dataset.CorrectionsVAL @ coefficients_values

    # Compute total quadratic correction
    if use_quad:
        coeff_outer_coeff = np.outer(coefficients_values, coefficients_values)
        summed_quad_corrections = dataset.HOcorrectionsVAL @ flatten(coeff_outer_coeff)
        summed_corrections += summed_quad_corrections

    # Sum of SM theory + SMEFT corrections
    corrected_theory = dataset.SMTheory + summed_corrections

    return corrected_theory
