# -*- coding: utf-8 -*-

"""
Module for the computation of chi-squared values
"""
import re

from . import compute_theory as pr

def chi2(diff, invcov):
    return diff @ invcov @ diff


def compute_chi2(dataset, coefficients_values, use_quad, use_replica, compute_per_dataset=False):
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
        compute_per_dataset: bool
            if True returns the :math:`\Chi^2` per dataset

    Returns
    -------
        chi2_total : float
            :math:`\Chi^2` value
        chi2_dict : dict
            reduced :math:`\Chi^2` value for each dataset
    """

    # compute theory prediction for each point in the dataset
    theory_predictions = pr.make_predictions(dataset, coefficients_values, use_quad)
    # compute experimental central values - theory
    if use_replica:
        mask = dataset.training_mask
        diff_tr = dataset.Replica[mask] - theory_predictions[mask]
        invcovmat_tr = dataset.InvCovMat[mask].T[mask]

        mask = ~dataset.training_mask
        diff_val = dataset.Replica[mask] - theory_predictions[mask]
        invcovmat_val = dataset.InvCovMat[mask].T[mask]

        return chi2(diff_tr, invcovmat_tr), chi2(diff_val, invcovmat_val)

        
    else:
        diff = dataset.Commondata - theory_predictions
        invcovmat = dataset.InvCovMat
        return chi2(diff, invcovmat)

    

    # # chi2 computation
    # chi2_vect = diff @ invcovmat
    # chi2_total = chi2_vect @ diff

    # # # chi2 per dataset
    # if compute_per_dataset:
    #     chi2_dict = {}
    #     cnt = 0
    #     for data_name, ndat in zip(dataset.ExpNames, dataset.NdataExp):
    #         chi2_dict[data_name] = float(
    #             chi2_vect[cnt : cnt + ndat] @ diff[cnt : cnt + ndat] / ndat
    #         )
    #         cnt += ndat
    #     return chi2_total, chi2_dict
    # return chi2_total
