# -*- coding: utf-8 -*-
import numpy as np

from smefit.analyze import pca
from smefit.coefficients import CoefficientManager
from smefit.loader import load_datasets

from .test_loader import commondata_path

operators_to_keep = np.array(["Op1", "Op2", "Op3", "Op4"])

dataset = load_datasets(
    commondata_path,
    datasets=["data_test5"],
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
        "Op4": {
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


def test_make_sym_matrix():

    vals = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    mat = pca.make_sym_matrix(vals, 3)
    np.testing.assert_equal(mat[:, :, 0], mat[:, :, 0].T)
    np.testing.assert_equal(mat[:, :, 1], mat[:, :, 1].T)
    np.testing.assert_equal(mat[:, :, 0], np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]]))
    np.testing.assert_equal(
        mat[:, :, 1], np.array([[7, 8, 9], [8, 10, 11], [9, 11, 12]])
    )

    # test diagonal
    diag_corr = np.diagonal(mat, axis1=0, axis2=1)
    np.testing.assert_equal(diag_corr, np.array([[1, 4, 6], [7, 10, 12]]))


def test_impose_constrain():

    updated_lincorr, updated_quadcorr = pca.impose_constrain(
        dataset, coefficients, update_quad=True
    )

    op1 = dataset.LinearCorrections[:, 0]
    op2 = dataset.LinearCorrections[:, 1]
    op3 = dataset.LinearCorrections[:, 2]
    op4 = dataset.LinearCorrections[:, 3]
    test_updated_lincorr = np.array([op1 + c13 * op3, op2 + c23 * op3, op4])
    np.testing.assert_equal(updated_lincorr.shape, (3, 2))
    np.testing.assert_equal(updated_lincorr, test_updated_lincorr)

    op1op1 = dataset.QuadraticCorrections[:, 0]
    op1op2 = dataset.QuadraticCorrections[:, 1]
    op1op3 = dataset.QuadraticCorrections[:, 2]
    op1op4 = dataset.QuadraticCorrections[:, 3]
    op2op2 = dataset.QuadraticCorrections[:, 4]
    op2op3 = dataset.QuadraticCorrections[:, 5]
    op2op4 = dataset.QuadraticCorrections[:, 6]
    op3op3 = dataset.QuadraticCorrections[:, 7]
    op3op4 = dataset.QuadraticCorrections[:, 8]
    op4op4 = dataset.QuadraticCorrections[:, 9]

    d1 = op1op1 + c13**2 * op3op3 + c13 * op1op3
    d2 = op2op2 + c23**2 * op3op3 + c23 * op2op3
    d12 = op1op2 + c13 * op2op3 + c23 * op1op3
    d14 = op1op4 + c13 * op3op4
    d24 = op2op4 + c23 * op3op4
    d4 = op4op4
    test_updated_quadcorr = pca.make_sym_matrix(
        np.array([d1, d12, d14, d2, d24, d4]).T, 3
    )
    np.testing.assert_equal(test_updated_quadcorr.shape, (3, 3, 2))
    np.testing.assert_allclose(updated_quadcorr, test_updated_quadcorr, rtol=1e-15)


def test_pca_eig():

    """Test the relation of SVD and normal eigenvalue decomposition."""
    pca_cal = pca.PcaCalculator(dataset, coefficients, latex_names=None)
    pca_cal.compute()

    new_LinearCorrections = pca.impose_constrain(dataset, coefficients)
    X = new_LinearCorrections @ dataset.InvCovMat @ new_LinearCorrections.T
    D, N = np.linalg.eig(X)
    S = pca_cal.SVs.values
    V = pca_cal.pc_matrix.values

    np.testing.assert_allclose(S**2, np.sort(D**2)[::-1], atol=1e-17)
    np.testing.assert_allclose(np.abs(V), np.abs(N), atol=1e-17)