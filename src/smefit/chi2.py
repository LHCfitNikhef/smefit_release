# -*- coding: utf-8 -*-

"""
Module for the computation of chi-squared values
"""
from . import compute_theory as pr


def compute_chi2(dataset, coefficients_values, use_quad, use_replica=False):
    r"""
    Compute the chi2
    Will need to be modified when implementing training validation split.

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
        chi2_total : float
            :math:`\Chi^2` value
    """

    # compute theory prediction for each point in the dataset
    theory_predictions = pr.make_predictions(dataset, coefficients_values, use_quad)

    # compute experimental central values - theory
    if use_replica:
        diff = dataset.Replica - theory_predictions
    else:
        diff = dataset.Commondata - theory_predictions

    invcovmat = dataset.InvCovMat
    return diff @ invcovmat @ diff
