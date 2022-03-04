# -*- coding: utf-8 -*-

"""
Module for the generation of theory predictions
"""
import numpy as np


def make_predictions(dataset, coeffs, labels, HOindex1, HOindex2):
    """
    Generate the corrected theory predictions for dataset `set`
    given a set of |SMEFT| coefficients `coeffs`. Optionally a specific
    operator may be selected with `iop`

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
        corrected_theory : numpy.ndarray
            SM + EFT theory predictions
    """

    # Compute total linear correction
    idx = np.where(dataset.CorrectionsKEYS == labels)[0]
    summed_corrections = dataset.CorrectionsVAL @ coeffs[idx]

    # Compute total quadratic correction
    if (HOindex1 is not None):
        idx1 = HOindex1
        idx2 = HOindex2
        coeffs_quad = coeffs[idx1] * coeffs[idx2]
        summed_quad_corrections = dataset.HOcorrectionsVAL @ coeffs_quad
        summed_corrections += summed_quad_corrections

    # Sum of SM theory + SMEFT corrections
    print(dataset.SMTheory)
    print(summed_corrections)
    corrected_theory = dataset.SMTheory + summed_corrections

    return corrected_theory
