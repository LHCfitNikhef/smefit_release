# -*- coding: utf-8 -*-

"""
Module for the generation of theory predictions
"""
import numpy as np


def make_predictions(dataset, coefficients, labels, HOindices):
    """
    Generate the corrected theory predictions for dataset
    given a set of |SMEFT| coefficients.

    Parameters
    ----------
        dataset : DataTuple
            dataset tuple
        coefficients : numpy.ndarray
            |EFT| corrections
        lables : list(str)
            list of coefficient to include
        HOindices: dict, None
            dictionary with HO corrections locations. None for linear fits
    Returns
    -------
        corrected_theory : numpy.ndarray
            SM + EFT theory predictions
    """

    # Compute total linear correction
    idx = np.where(dataset.CorrectionsKEYS == labels)[0]
    summed_corrections = dataset.CorrectionsVAL @ coefficients[idx]

    # Compute total quadratic correction
    if HOindices is not None:
        coefficients = coefficients[HOindices[1]] * coefficients[HOindices[2]]
        summed_quad_corrections = dataset.HOcorrectionsVAL @ coefficients
        summed_corrections += summed_quad_corrections

    # Sum of SM theory + SMEFT corrections
    corrected_theory = dataset.SMTheory + summed_corrections

    return corrected_theory
