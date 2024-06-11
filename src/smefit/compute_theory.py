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
    dataset, coefficients_values, use_quad, use_multiplicative_prescription, rgemat=None
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
    if rgemat is not None:
        summed_corrections = jnp.einsum(
            "ij,jk,k->i", dataset.LinearCorrections, rgemat.values, coefficients_values
        )
    else:
        summed_corrections = jnp.einsum(
            "ij,j->i", dataset.LinearCorrections, coefficients_values
        )

    # Compute total quadratic correction
    if use_quad:
        if rgemat is not None:
            ext_coeffs = jnp.einsum("ij,j->i", rgemat.values, coefficients_values)
            coeff_outer_coeff = jnp.outer(ext_coeffs, ext_coeffs)
        else:
            coeff_outer_coeff = jnp.outer(coefficients_values, coefficients_values)
        # note @ is slower when running with mpiexec
        summed_quad_corrections = jnp.einsum(
            "ij,j->i", dataset.QuadraticCorrections, flatten(coeff_outer_coeff)
        )
        summed_corrections += summed_quad_corrections

    # Sum of SM theory + SMEFT corrections
    if use_multiplicative_prescription:
        corrected_theory = dataset.SMTheory * (1.0 + summed_corrections)
    else:
        corrected_theory = dataset.SMTheory + summed_corrections

    return corrected_theory
