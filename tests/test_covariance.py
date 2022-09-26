# -*- coding: utf-8 -*-
"""Test covmat module"""
import numpy as np
import pandas as pd

from smefit import covmat

stat = np.ones(3)
sys = np.asarray([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
covmat_test = np.diag(stat) + sys @ sys.T
sys_names = ["CORR", "CORR", "CORR", "CORR"]
sys_dataframe = pd.DataFrame(data=sys, columns=sys_names)


def test_construct_covmat():
    np.testing.assert_allclose(
        covmat.construct_covmat(stat, sys_dataframe), covmat_test
    )


first_covmat = np.asarray([[1, 2], [3, 4]])
second_covmat = np.asarray([[5, 6, 7], [8, 9, 10], [11, 12, 13]])
covmat_list = [first_covmat, second_covmat]
big_covmat = np.asarray(
    [
        [1, 2, 0, 0, 0],
        [3, 4, 0, 0, 0],
        [0, 0, 5, 6, 7],
        [0, 0, 8, 9, 10],
        [0, 0, 11, 12, 13],
    ]
)
