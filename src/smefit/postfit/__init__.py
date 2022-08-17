# -*- coding: utf-8 -*-
import json
import pathlib
import shutil

import numpy as np
import yaml

from ..log import logging

_logger = logging.getLogger(__name__)


class Postfit:
    def __init__(self, result_path, result_ID, chi2_threshold):

        self.results_folder = pathlib.Path(result_path).absolute() / result_ID
        self.finished_replicas = 0
        for name in self.results_folder.iterdir():
            if "replica_" in str(name):
                self.finished_replicas += 1

        self.chi2_threshold = chi2_threshold
        self.not_completed = False

    @classmethod
    def from_file(cls, runcard_folder, run_card_name):

        # load file
        runcard_folder = pathlib.Path(runcard_folder)
        with open(runcard_folder / f"{run_card_name}.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # set result ID to runcard name by default
        result_ID = config.get("result_ID", run_card_name)

        # get the chi2_threshold, by default is 3.0
        if "chi2_threshold" not in config:
            _logger.warning("chi2_threshold not set, using default value of 3.0")
        chi2_threshold = config.get("chi2_threshold", 3.0)

        return cls(config["result_path"], result_ID, chi2_threshold)

    def save(self, nrep):

        postfit_res = []
        chi2_list = []

        if nrep > self.finished_replicas:
            raise ValueError(f"Only {self.finished_replicas} available")

        for rep in range(1, self.finished_replicas + 1):
            if len(postfit_res) == nrep:
                break
            rep_res = []
            with open(
                self.results_folder / f"replica_{rep}/coefficients_rep_{rep}.json",
                encoding="utf-8",
            ) as f:
                res = json.load(f)

            if len(postfit_res) == 0:
                coeffs = res.keys()

            if res["chi2"] > self.chi2_threshold:
                _logger.warning(f"Discarding replica: {rep}")
                continue

            chi2_list.append(res["chi2"])
            del res["chi2"]
            for coeff in res:
                rep_res.append(res[coeff])

            postfit_res.append(rep_res)

        _logger.info(f"Chi2 average : {np.mean(chi2_list)}")
        if len(postfit_res) < nrep:
            _logger.warning(
                f"Only {len(postfit_res)} replicas pass postfit, please run some more"
            )
            self.not_completed = True

        else:

            posterior = {}
            for i, c in enumerate(coeffs):
                posterior[c] = list(np.array(postfit_res).T[i, :])

            with open(
                self.results_folder / "posterior.json", "w", encoding="utf-8"
            ) as f:
                json.dump(posterior, f)

    def clean(self):
        if self.not_completed:
            _logger.warning("Cleaning not done, since you don't have enough replicas.")
            return
        for replica_folder in self.results_folder.iterdir():
            if "replica_" in str(replica_folder):
                shutil.rmtree(replica_folder)
