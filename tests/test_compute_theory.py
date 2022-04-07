# -*- coding: utf-8 -*-

"""Test compute_theory and chi2 module"""
import numpy as np

from smefit import loader
from smefit import compute_theory
from smefit import chi2

matrix = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
flatten_matrix = np.asarray([1, 2, 3, 5, 6, 9])


def test_flatten():
    np.testing.assert_allclose(compute_theory.flatten(matrix), flatten_matrix)


exp_data = np.asarray([1, 2, 3, 4])
sm_theory = np.asarray([1, 1, 1, 1])
operators_names = np.asarray(["Op1", "Op2"])
lin_corr_values = np.asarray([[1, 2], [1, 2], [1, 2], [1, 2]])  # Op1, Op2
quad_corr_values = np.asarray(
    [[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]]
)  # Op1^2, Op1*Op2, Op2^2
exp_name = np.asarray(["exp1"])
n_data_exp = np.asarray([4])
covmat = np.diag(np.ones(4))

wilson_coeff_values = np.asarray([0.5, 0.6])
linear_term = lin_corr_values @ wilson_coeff_values
quadratic_term = quad_corr_values @ compute_theory.flatten(
    np.outer(wilson_coeff_values, wilson_coeff_values)
)

corrected_linear = sm_theory + linear_term
corrected_quadratic = sm_theory + linear_term + quadratic_term

diff = corrected_quadratic - exp_data
chi2_test = diff @ np.linalg.inv(covmat) @ diff

dataset = loader.DataTuple(
    exp_data,
    sm_theory,
    operators_names,
    lin_corr_values,
    quad_corr_values,
    exp_name,
    n_data_exp,
    np.linalg.inv(covmat),
)


def test_make_predictions():
    np.testing.assert_allclose(
        compute_theory.make_predictions(dataset, wilson_coeff_values, False),
        corrected_linear,
    )
    np.testing.assert_allclose(
        compute_theory.make_predictions(dataset, wilson_coeff_values, True),
        corrected_quadratic,
    )


def test_compute_chi2():
    np.testing.assert_allclose(
        chi2.compute_chi2(dataset, wilson_coeff_values, True), chi2_test
    )
