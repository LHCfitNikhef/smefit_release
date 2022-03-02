"""
Module for the computation of chi-squared values
"""
import numpy as np
from . import compute_theory as pr


def compute_chi2(config, dataset, coeffs, labels):
    """
    Compute the components for the chi2

    Here we also perform the cross-validation splitting at the level of the residuals,
    so as to prevent singular covariances matrices.

    A mask is applied to each experiment and if the dataset has only 1 datapoint
    it is placed in the training set

    Returns the theory - exp vector and the inverse cov mat * (theory - exp)

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
        diff : numpy.ndarray
            (theory - exp) vector
        diff_mask : numpy.array
            (theory - exp) vector with mask, only for cross validation
        covmatdiff : numpy.ndarray
            inverse cov mat * (theory - exp)
        covmatdiff_mask : numpy.ndarray
            inverse cov mat * (theory - exp) with mask, only for cross validation
    """

    # compute theory prediction for each point in the dataset
    theory_predictions = pr.make_predictions(config, dataset, coeffs, labels)
    # get central values of experimental points
    dat = dataset.Commondata
    # compute data - theory
    diff = dat - theory_predictions

    # The chi2 computation
    covmat_inv = np.linalg.inv(dataset.CovMat)
    # Multiply cov^-1 * diff but don't sum
    covmatdiff = np.einsum("ij,j->i", covmat_inv, diff)

    return diff, covmatdiff


def compute_total_chi2(config, datasets, coefficients, labels):
    """
    Function to compute total central chi2 for all datasets
    assuming no cross-correlations. Returns the total chi2. 

    Parameters
    ----------
        config : dict
            configuration dictionary
        dataset : DataTuple
            dataset tuple
        coefficients : numpy.ndarray
            coefficients list
        lables : list(str)
            labels list

    Returns
    -------
        chi2_total : numpy.ndarray
            chi2 values
        dof : int
            sum of number of datapoints
    """
    diff_total = []
    covmatdiff_total = []

    Ndat = []

    # Read in the results from the chi2 function
    diff,  covmatdiff = compute_chi2(
        config, datasets, coefficients, labels
    )

    diff_total.extend(diff)
    covmatdiff_total.extend(covmatdiff)

    Ndat.append(len(datasets.Commondata))

    # Return the total chi2 (which is in the training set)
    chi2_total = np.einsum("j,j->", diff_total, covmatdiff_total)

    
    return chi2_total, np.sum(Ndat), 0.0
