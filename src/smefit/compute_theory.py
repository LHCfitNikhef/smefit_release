# -*- coding: utf-8 -*-

"""
Module for the generation of theory predictions
"""
import jax
import jax.numpy as jnp
import numpy as np


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
        use_multiplicative_prescription: bool
            if True add the |EFT| contribution as a k-factor
        rgemat: numpy.ndarray
            solution matrix of the RGE
    Returns
    -------
        corrected_theory : numpy.ndarray
            SM + EFT theory predictions
    """

    coefficients_values = jnp.array(coefficients_values)

    # Compute total linear correction
    # if rgemat is not None:
    #     # Check if rgemat comes from a dynamic scale
    #     # otherwise it is a single RGEmatrix
    #     if len(rgemat.shape) == 3:
    #         wcs = jnp.einsum("nij,j->ni", rgemat, coefficients_values)
    #         summed_corrections = jnp.einsum("ij,ij->i", dataset.LinearCorrections, wcs)

    #     else:
    #         summed_corrections = jnp.einsum(
    #             "ij,jk,k->i",
    #             dataset.LinearCorrections,
    #             rgemat,
    #             coefficients_values,
    #         )
    # else:
    summed_corrections = jnp.einsum(
        "ij,j->i", dataset.LinearCorrections, coefficients_values
    )

    # Compute total quadratic correction
    # if use_quad:
    #     if rgemat is not None:
    #         # check that rgemat is a 3D array
    #         if len(rgemat.shape) == 3:
    #             # do outer product on previously computed wcs
    #             coeff_outer_coeff = jnp.einsum("ni,nj->nij", wcs, wcs)
    #             coeff_outer_coeff_flat = jax.vmap(flatten)(coeff_outer_coeff)
    #             summed_quad_corrections = jnp.einsum(
    #                 "ij,ij->i", dataset.QuadraticCorrections, coeff_outer_coeff_flat
    #             )

    #         else:
    #             ext_coeffs = jnp.einsum("ij,j->i", rgemat, coefficients_values)
    #             coeff_outer_coeff = jnp.outer(ext_coeffs, ext_coeffs)
    #             summed_quad_corrections = jnp.einsum(
    #                 "ij,j->i", dataset.QuadraticCorrections, flatten(coeff_outer_coeff)
    #             )
    #     else:
    #         coeff_outer_coeff = jnp.outer(coefficients_values, coefficients_values)
    #         # note @ is slower when running with mpiexec
    #         summed_quad_corrections = jnp.einsum(
    #             "ij,j->i", dataset.QuadraticCorrections, flatten(coeff_outer_coeff)
    #         )
    #     summed_corrections += summed_quad_corrections

    # Sum of SM theory + SMEFT corrections
    if use_multiplicative_prescription:
        corrected_theory = dataset.SMTheory * (1.0 + summed_corrections)
    else:
        corrected_theory = dataset.SMTheory + summed_corrections

    return corrected_theory
