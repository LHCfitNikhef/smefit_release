# -*- coding: utf-8 -*-
import json
import pathlib

import numpy as np
from ml4eft.limits.optimize_ns import Optimize


class MLObservables:
    def __init__(self, coefficients):

        self.dir_abs = pathlib.Path(__file__).parent.resolve()
        self.coefficients = coefficients
        self.ml_likelihood, self.idxs = self.load_ml_likelihood()
        # self.ml_likelihood = self.load_ml_likelihood()

    def load_ml_likelihood(self):

        ml_runcard = pathlib.Path("NS_run_card_zhllbb.json")

        with open(self.dir_abs / ml_runcard) as json_data:
            config = json.load(json_data)

        # coeffs = np.array(["cHW", "cHWB", "cHj1", "cHj3", "cHu", "cHd", "cbHRe"])

        coeffs_dict = {
            "OpW": "cHW",
            "OpWB": "cHWB",
            "OpqMi": "cHj1",
            "O3pq": "cHj3",
            "Opui": "cHu",
            "Opdi": "cHd",
            "Obp": "cbHRe",
        }

        coeffs = np.array([coeffs_dict[coeff] for coeff in self.coefficients.name])

        runner = Optimize(config, coeff=coeffs)

        idx0 = np.argwhere(self.coefficients.name == "OpqMi").flatten()[0]
        idx1 = np.argwhere(self.coefficients.name == "O3pq").flatten()[0]

        return runner.log_like_binned, [idx0, idx1]
        # runner = Optimize(config, coeff=coeffs_dict[self.coefficients.name[0]])
        # return runner.log_like_nn, [idx0, idx1]

    def evaluate_likelihood(self, coefficient_values):

        # remap cHj1 and cHj3. coefficient_values follows the smefit convention
        # coefficient_values[self.idxs[1]] = coefficient_values[self.idxs[0]] + coefficient_values[self.idxs[1]]
        coefficient_values[self.idxs[0]] = (
            coefficient_values[self.idxs[0]] + coefficient_values[self.idxs[1]]
        )

        return self.ml_likelihood(coefficient_values)
