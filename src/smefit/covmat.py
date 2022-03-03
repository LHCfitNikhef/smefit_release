# -*- coding: utf-8 -*-

"""
Module containing computation of covariance matrix
"""
import numpy as np
import pandas as pd


def construct_covmat(stat_errors: np.array, sys_errors: pd.DataFrame):
    """
    This function is taken from: https://github.com/NNPDF/nnpdf/tree/master/validphys2/src/validphys/covmats_utils.py

    Basic function to construct a covariance matrix (covmat), given the
    statistical error and a dataframe of systematics.
    Errors with name UNCORR or THEORYUNCORR are added in quadrature with
    the statistical error to the diagonal of the covmat.
    Other systematics are treated as correlated; their covmat contribution is
    found by multiplying them by their transpose.

    Parameters
    ----------
    stat_errors: numpy.ndarray
        a 1-D array of statistical uncertainties
    sys_errors: pandas.DataFrame
        a dataframe with shape (N_data * N_sys) and systematic name as the
        column headers. The uncertainties should be in the same units as the
        data.

    Returns
    -------
        cov_mat: numpy.ndarray
            Covariance matrix

    Notes
    -----
    This function doesn't contain any logic to ignore certain contributions to
    the covmat, if you wanted to not include a particular systematic/set of
    systematics i.e all uncertainties with MULT errors, then filter those out
    of ``sys_errors`` before passing that to this function.

    """
    diagonal = stat_errors**2

    is_uncorr = sys_errors.columns.isin(("UNCORR", "THEORYUNCORR"))
    diagonal += (sys_errors.loc[:, is_uncorr].to_numpy() ** 2).sum(axis=1)

    corr_sys_mat = sys_errors.loc[:, ~is_uncorr].to_numpy()
    return np.diag(diagonal) + corr_sys_mat @ corr_sys_mat.T


def build_large_covmat(ndata, chi2_covmat, n_data_exp):
    """
    Build large covariance matrix (individual datsets are on diagonal, no cross-correlations)

    Parameters
    ----------
        ndata : int
            total number of datapoints
        n_data_exp: list
            list of number of data per experiment
        chi2_covmat: np.ndarray
            chi 2 covariance matrix

    Returns
    -------
        covmat_array: np.ndarray
            total experimental covariance matrix
    """
    covmat_array = np.zeros((ndata, ndata))
    cnt = 0

    for i, nexpdata in enumerate(n_data_exp):
        for j in range(nexpdata):
            for k in range(nexpdata):
                covmat_array[cnt + j, cnt + k] = chi2_covmat[i][j][k]
        cnt += nexpdata
    return covmat_array
