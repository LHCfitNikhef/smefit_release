# -*- coding: utf-8 -*-

"""
Module for the computation of chi-squared values
"""
import numpy as np

from . import compute_theory as pr


def compute_chi2(dataset, coefficients, nho_indices, ho_indices):
    """
    Compute the chi2
    Will need to be modified when implementing training validation split.

    Parameters
    ----------
        config : dict
            configuration dictionary
        dataset : DataTuple
            dataset tuple
        coefficients : numpy.ndarray
            coefficients list
        nho_indices : list
            list of |NHO| corrections locations
        ho_indices: dict, None
            dictionary with HO corrections locations. None for linear fits


    Returns
    -------
        chi2_total : numpy.ndarray
            chi2 values
    """

    # compute theory prediction for each point in the dataset
    theory_predictions = pr.make_predictions(
        dataset, coefficients, nho_indices, ho_indices
    )
    # get central values of experimental points
    dat = dataset.Commondata

    # compute data - theory
    diff = dat - theory_predictions

    # The chi2 computation
    cov_mat_inv = np.linalg.inv(dataset.CovMat)
    # TODO einsum is slower, consider to remove it, for simple operations
    # Multiply cov^-1 * diff
    cov_mat_diff = np.einsum("ij,j->i", cov_mat_inv, diff)
    # Multiply diff * (cov^-1 * diff) to get chi2
    chi2_total = np.einsum("j,j->", diff, cov_mat_diff)

    return chi2_total
