# -*- coding: utf-8 -*-

"""
Module for the generation of theory predictions
"""


def make_predictions(dataset, coefficients, lin_indices, quad_indices):
    """
    Generate the corrected theory predictions for dataset
    given a set of |SMEFT| coefficients.

    Parameters
    ----------
        dataset : DataTuple
            dataset tuple
        coefficients : numpy.ndarray
            |EFT| corrections
        lin_indices : list
            list of |NHO| corrections locations
        quad_indices: dict, None
            dictionary of |HO| corrections locations. None for linear fits
    Returns
    -------
        corrected_theory : numpy.ndarray
            SM + EFT theory predictions
    """

    # Compute total linear correction
    summed_corrections = dataset.CorrectionsVAL @ coefficients[lin_indices]

    # Compute total quadratic correction
    if quad_indices is not None:
        coefficients = coefficients[quad_indices[1]] * coefficients[quad_indices[2]]
        summed_quad_corrections = dataset.HOcorrectionsVAL @ coefficients
        summed_corrections += summed_quad_corrections

    # Sum of SM theory + SMEFT corrections
    corrected_theory = dataset.SMTheory + summed_corrections

    return corrected_theory
