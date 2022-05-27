# -*- coding: utf-8 -*-
"""Test loader module"""
import pathlib

import numpy as np

from smefit.loader import load_datasets

commondata_path = pathlib.Path(__file__).parents[0] / "fake_data"


exp_test = {"data": [1, 2]}

theory_test = {
    "best_sm": [1.0, 2.0],
    "theory_cov": [[0.1, 0.2], [0.2, 0.3]],
    "LO": {
        "Op1": [0.1, 0.2],
        "Op3": [0.5, 0.6],
        "Op2": [0.3, 0.4],
        "Op1*Op1": [0.7, 0.8],
        "Op2*Op2": [0.9, 0.11],
        "Op2*Op1": [0.12, 0.13],
        "Op2*Op3": [0.14, 0.15],
    },
    "NLO": {
        "Op1": [0.1, 0.2],
        "Op3": [0.5, 0.6],
        "Op2": [0.3, 0.4],
        "Op1*Op1": [0.7, 0.8],
        "Op2*Op2": [0.9, 0.11],
        "Op2*Op1": [0.12, 0.13],
        "Op2*Op3": [0.14, 0.15],
    },
}


def test_load_datasets():
    operators_to_keep = np.array(["Op1", "Op2"])
    for use_quad in [True, False]:
        for order in ["LO", "NLO"]:
            loaded_tuple = load_datasets(
                commondata_path,
                datasets=["data_test"],
                operators_to_keep=operators_to_keep,
                order=order,
                use_quad=use_quad,
                use_theory_covmat=True,
                theory_path=commondata_path,
                rot_to_fit_basis=None,
            )

            lin_corr = []
            for op in operators_to_keep:
                lin_corr.append(theory_test[order][op])

            np.testing.assert_equal(loaded_tuple.Commondata, exp_test["data"])
            np.testing.assert_equal(loaded_tuple.SMTheory, theory_test["best_sm"])
            np.testing.assert_equal(loaded_tuple.OperatorsNames, operators_to_keep)
            np.testing.assert_equal(loaded_tuple.ExpNames, ["data_test"])
            np.testing.assert_equal(loaded_tuple.NdataExp, [2])
            np.testing.assert_equal(loaded_tuple.LinearCorrections.T, lin_corr)

            if use_quad:
                quad_corr = [
                    theory_test[order]["Op1*Op1"],
                    theory_test[order]["Op2*Op1"],
                    theory_test[order]["Op2*Op2"],
                ]
                np.testing.assert_equal(loaded_tuple.QuadraticCorrections.T, quad_corr)


def test_operator_correction_sorted():
    operators_to_keep = np.array(["Op1", "Op3", "Op2"])
    for order in ["LO", "NLO"]:
        loaded_tuple = load_datasets(
            commondata_path,
            datasets=["data_test"],
            operators_to_keep=operators_to_keep,
            order=order,
            use_quad=True,
            use_theory_covmat=True,
            theory_path=commondata_path,
            rot_to_fit_basis=None,
        )

        lin_corr = [
            theory_test[order]["Op1"],
            theory_test[order]["Op2"],
            theory_test[order]["Op3"],
        ]

        np.testing.assert_equal(loaded_tuple.LinearCorrections.T, lin_corr)

        quad_corr = [
            theory_test[order]["Op1*Op1"],
            theory_test[order]["Op2*Op1"],
            np.zeros(2),  # Op1*Op3
            theory_test[order]["Op2*Op2"],
            theory_test[order]["Op2*Op3"],
            np.zeros(2),  # Op3*Op3
        ]
        np.testing.assert_equal(loaded_tuple.QuadraticCorrections.T, quad_corr)
