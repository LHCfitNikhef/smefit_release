import json
import yaml
import pathlib
import os, os.path
import numpy as np


class Postfit:
    def __init__(self, run_card, runcard_folder):

        self.results_folder = (
            pathlib.Path(run_card["result_path"]) / run_card["result_ID"]
        )
        self.finished_replicas = len(
            [name for name in os.listdir(self.results_folder) if "replica_" in name]
        )

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

        return cls(config, runcard_folder)

    def save(self, nrep):

        postfit_res = []

        if nrep > self.finished_replicas:
            print(f"Only {self.finished_replicas} available")
            exit(1)

        for rep in range(1, nrep + 1):
            rep_res = []
            with open(
                self.results_folder / f"replica_{rep}/coefficients_rep_{rep}.json"
            ) as f:
                res = json.load(f)

            if len(postfit_res) == 0:
                coeffs = res.keys()

            if res["chi2"] > 3.0:
                continue

            del res["chi2"]
            for coeff in res.keys():
                rep_res.append(res[coeff])

            postfit_res.append(rep_res)

        if len(postfit_res) < nrep:
            print(
                f"Only {len(postfit_res)} replicas pass postfit, please run some more"
            )

        else:

            posterior = {}
            for i, c in enumerate(coeffs):
                posterior[c] = list(np.array(postfit_res).T[i, :])

            with open(
                self.results_folder / "posterior_mc.json", "w", encoding="utf-8"
            ) as f:
                json.dump(posterior, f)
