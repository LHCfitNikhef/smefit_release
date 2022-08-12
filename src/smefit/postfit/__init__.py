# -*- coding: utf-8 -*-
import json
import pathlib
import shutil

import numpy as np
import yaml

from ..log import logging

_logger = logging.getLogger(__name__)


class Postfit:
    def __init__(self, run_card, chi2_threshold=3.0):

        self.results_folder = (
            pathlib.Path(run_card["result_path"]).absolute() / run_card["result_ID"]
        )
        self.finished_replicas = 0
        for name in self.results_folder.iterdir():
            if "replica_" in str(name):
                self.finished_replicas += 1

        self.chi2_threshold = chi2_threshold

    @classmethod
    def from_file(cls, runcard_folder, run_card_name):

        config = {}
        # load file
        runcard_folder = pathlib.Path(runcard_folder)
        with open(runcard_folder / f"{run_card_name}.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        config["runcard_name"] = run_card_name
        # set result ID to runcard name by default
        if "result_ID" not in config:
            config["result_ID"] = run_card_name

        return cls(config)

    def save(self, nrep):

        postfit_res = []

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

            del res["chi2"]
            for coeff in res:
                rep_res.append(res[coeff])

            postfit_res.append(rep_res)

        if len(postfit_res) < nrep:
            _logger.warning(
                f"Only {len(postfit_res)} replicas pass postfit, please run some more"
            )

        else:

            posterior = {}
            for i, c in enumerate(coeffs):
                posterior[c] = list(np.array(postfit_res).T[i, :])

            with open(
                self.results_folder / "posterior.json", "w", encoding="utf-8"
            ) as f:
                json.dump(posterior, f)

    def clean(self):
        for replica_folder in self.results_folder.iterdir():
            if "replica_" in str(replica_folder):
                shutil.rmtree(replica_folder)
