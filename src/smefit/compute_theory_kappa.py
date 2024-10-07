# -*- coding: utf-8 -*-

"""
Module for the generation of theory predictions
"""
import numpy as np
import jax.numpy as jnp


def flatten(quad_mat, axis=0):
    """
    Delete lower triangular part of a quadratic matrix
    and flatten it into an array

    Parameters
    ----------
        quad_mat: numpy.ndarray
            tensor to flatten
        axis: int
            axis along which the triangular part is selected
    """
    size = quad_mat.shape[axis]
    return quad_mat[np.triu_indices(size)]


def make_predictions(
    dataset, coefficients_values, use_quad, use_multiplicative_prescription
):
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

    coefficients_values = jnp.array(coefficients_values)

    # Compute total linear correction
    # note @ is slower when running with mpiexec
    summed_corrections = jnp.einsum(
        "ij,j->i", dataset.LinearCorrections, coefficients_values
    )

    # Compute total quadratic correction
    coeff2=jnp.power(coefficients_values,2)
    """
    if use_quad: INSTEAD OF USE_QUAD we should put a kappa_frm as label, that would also enter the run_card
    """
    #each kappa is entering squared    
    coeff_outer_coeff = jnp.outer(coeff2, coeff2)
    # note @ is slower when running with mpiexec
    summed_quad_corrections = jnp.einsum(
            "ij,j->i", dataset.QuadraticCorrections, flatten(coeff_outer_coeff)
        )
    summed_corrections += summed_quad_corrections

    # Sum of SM theory + SMEFT corrections
        # define the total k_H
    k_weights=jnp.array([0.57,0.06,0.029,0.22,0.027,0.086,0.0023,0.0016,0.00022,0.]) #If this is going to become a module we should put this array somewhere else
    k_H=jnp.sum(k_weights*coeff2)
    return summed_corrections/k_H 
    
