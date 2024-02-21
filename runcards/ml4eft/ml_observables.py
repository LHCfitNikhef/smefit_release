# -*- coding: utf-8 -*-
import json
import pathlib

import numpy as np
from ml4eft.limits.optimize_ns import Optimize


class MLObservables:
    def __init__(self, coefficients):

        self.dir_abs = pathlib.Path(__file__).parent.resolve()
        self.ml_likelihood = self.load_ml_likelihood()

    def load_ml_likelihood(self):

        ml_runcard = pathlib.Path("NS_run_card_zhllbb.json")

        with open(self.dir_abs / ml_runcard) as json_data:
            config = json.load(json_data)

        coeffs = np.array(["cHW", "cHWB", "cHu", "cHd", "cbHRe"])
        runner = Optimize(config, coeff=coeffs)
        return runner.log_like_nn

    def evaluate_likelihood(self, coefficient_values):

        return self.ml_likelihood(coefficient_values)
