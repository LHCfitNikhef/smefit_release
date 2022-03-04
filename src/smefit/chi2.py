# -*- coding: utf-8 -*-

"""
Module for the computation of chi-squared values
"""
import numpy as np

from . import compute_theory as pr


def compute_chi2(dataset, coeffs, labels, HOindex1, HOindex2):
    """
    Compute the chi2
    Will need to be modified when implementing training validation split.

    Parameters
    ----------
        config : dict
            configuration dictionary
        dataset : DataTuple
            dataset tuple
        coeffs : numpy.ndarray
            coefficients list
        lables : list(str)
            labels list

    Returns
    -------
        chi2_total : numpy.ndarray
            chi2 values
        dof : int
            number of datapoints

    """

    # compute theory prediction for each point in the dataset
    theory_predictions = pr.make_predictions(dataset, coeffs, labels, HOindex1, HOindex2)
    # get central values of experimental points
    dat = dataset.Commondata
    Ndat = len(dataset.Commondata)

    # compute data - theory
    diff = dat - theory_predictions

    # The chi2 computation
    covmat_inv = np.linalg.inv(dataset.CovMat)
    # TODO einsum is slower, consider to remove it, for simple operations
    # Multiply cov^-1 * diff
    covmatdiff = np.einsum("ij,j->i", covmat_inv, diff)
    # Multiply diff * (cov^-1 * diff) to get chi2
    chi2_total = np.einsum("j,j->", diff, covmatdiff)

    return chi2_total, Ndat
