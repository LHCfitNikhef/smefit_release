# -*- coding: utf-8 -*-

"""
Module for the generation of theory predictions
"""


def make_predictions(dataset, coefficients, nho_indices, ho_indices):
    """
    Generate the corrected theory predictions for dataset
    given a set of |SMEFT| coefficients.

    Parameters
    ----------
        dataset : DataTuple
            dataset tuple
        coefficients : numpy.ndarray
            |EFT| corrections
        nho_indices : list
            list of |NHO| corrections locations
        ho_indices: dict, None
            dictionary of |HO| corrections locations. None for linear fits
    Returns
    -------
        corrected_theory : numpy.ndarray
            SM + EFT theory predictions
    """

    # Compute total linear correction
    summed_corrections = dataset.CorrectionsVAL @ coefficients[nho_indices]

    # Compute total quadratic correction
    if ho_indices is not None:
        coefficients = coefficients[ho_indices[1]] * coefficients[ho_indices[2]]
        summed_quad_corrections = dataset.HOcorrectionsVAL @ coefficients
        summed_corrections += summed_quad_corrections

    # Sum of SM theory + SMEFT corrections
    corrected_theory = dataset.SMTheory + summed_corrections

    return corrected_theory
