# -*- coding: utf-8 -*-

"""
Module for the generation of theory predictions
"""
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


def string_to_function(expressions,param_dict): 
    """Substitute to the theory card expression the numerical values, keeping also into account external
    defined parameters
    
    Parameters
    ----------
    expressions: the array of theory card dictionary keys,
    param_dict: dictionary with the values of the point in the parameter space tried and the external parameters numerical value
     """
    return [eval(e,param_dict) for e in expressions ]

def make_predictions(
    dataset, coefficients_values, use_quad, use_multiplicative_prescription, poly_mode
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
    Returns
    -------
        corrected_theory : numpy.ndarray
            SM + EFT theory predictions
    """
    if poly_mode:
        summed_corrections=[]
        #create dictionary of operator and the point tried
        dict_point=dict(zip(dataset.OperatorsNames,coefficients_values))
        #extend dictionary evaluating external parametrs
        for ext,val in dataset.ExternalCoefficients.items():
            dict_point[ext]=eval(val,{},dict_point)

        #OperatorsDictionary[][1] are the labels,  [][2] are the numerical coefficients
        for op_dict in dataset.OperatorsDictionary:
            op_dict_num_values=string_to_function(op_dict[0],dict_point)
            summed_corrections.extend(jnp.dot(jnp.array(op_dict_num_values),jnp.array(op_dict[1])))
        #construct the theory precition 
        if use_multiplicative_prescription:
            corrected_theory = jnp.array(dataset.SMTheory)*(1.0 + jnp.array(summed_corrections))
        else:
            corrected_theory = jnp.array(dataset.SMTheory) + jnp.array(summed_corrections)  
        return corrected_theory
        
    else:

        coefficients_values = jnp.array(coefficients_values)

        summed_corrections = jnp.einsum(
            "ij,j->i", dataset.LinearCorrections, coefficients_values
        )
    
        # Compute total quadratic correction
        if use_quad:
            summed_quad_corrections = jnp.einsum(
            "ijk,j,k -> i",
            dataset.QuadraticCorrections,
            coefficients_values,
            coefficients_values,
        )
            summed_corrections += summed_quad_corrections
    
        # Sum of SM theory + SMEFT corrections
        if use_multiplicative_prescription:
            corrected_theory = dataset.SMTheory * (1.0 + summed_corrections)
        else:
            corrected_theory = dataset.SMTheory + summed_corrections
    
        return corrected_theory

