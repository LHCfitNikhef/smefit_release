# -*- coding: utf-8 -*-
import numpy as np

from smefit.analyze import fisher, pca
from smefit.coefficients import CoefficientManager
from smefit.loader import load_datasets

from .test_loader import commondata_path


def test_make_sym_matrix():

    vals = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    mat = fisher.make_sym_matrix(vals, 3)
    np.testing.assert_equal(mat[0], mat[0].T)
    np.testing.assert_equal(mat[1], mat[1].T)
    np.testing.assert_equal(mat[0], np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]]))
    np.testing.assert_equal(mat[1], np.array([[7, 8, 9], [8, 10, 11], [9, 11, 12]]))

    # test diagonal
    diag_corr = np.diagonal(mat, axis1=1, axis2=2)
    np.testing.assert_equal(diag_corr, np.array([[1, 4, 6], [7, 10, 12]]))


def test_impose_constrain():

    operators_to_keep = np.array(["Op1", "Op2", "Op3"])
    datasets = ["data_test1"]

    dataset = load_datasets(
        commondata_path,
        datasets=datasets,
        operators_to_keep=operators_to_keep,
        order="NLO",
        use_quad=True,
        use_theory_covmat=True,
        use_t0=False,
        use_multiplicative_prescription=False,
        theory_path=commondata_path,
        rot_to_fit_basis=None,
    )
    c23 = 0.1
    c13 = -0.2
    coefficients = CoefficientManager.from_dict(
        {
            "Op1": {
                "min": -1,
                "max": 1,
            },
            "Op2": {
                "min": -3,
                "max": 1,
            },
            "Op3": {  # fixed to 0.1 * Op2 - 0.2 * Op1
                "constrain": [
                    {"Op2": c23},
                    {"Op1": c13},
                ],
                "min": -5,
                "max": 1,
            },
        }
    )
    updated_lincorr, updated_quadcorr = pca.impose_constrain(
        dataset, coefficients, update_quad=True
    )

    op1 = dataset.LinearCorrections[:, 0]
    op2 = dataset.LinearCorrections[:, 1]
    op3 = dataset.LinearCorrections[:, 2]
    test_updated_lincorr = np.array([op1 + c13 * op3, op2 + c23 * op3])
    np.testing.assert_equal(updated_lincorr.shape, (2, 2))
    np.testing.assert_equal(updated_lincorr, test_updated_lincorr)

    op1op1 = dataset.QuadraticCorrections[:, 0]
    op1op2 = dataset.QuadraticCorrections[:, 1]
    op1op3 = dataset.QuadraticCorrections[:, 2]
    op2op2 = dataset.QuadraticCorrections[:, 3]
    op2op3 = dataset.QuadraticCorrections[:, 4]
    op3op3 = dataset.QuadraticCorrections[:, 5]

    d1 = op1op1 + c13**2 * op3op3 + c13 * op1op3
    d2 = op2op2 + c23**2 * op3op3 + c23 * op2op3
    d12 = op1op2 + c13 * op2op3 + c23 * op1op3
    np.testing.assert_equal(updated_quadcorr.shape, (2 * 3 / 2, 2))
    np.testing.assert_allclose(updated_quadcorr, [d1, d12, d2], rtol=1e-15)
