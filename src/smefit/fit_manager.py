# -*- coding: utf-8 -*-
import json
import pickle

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from rich.progress import track

from .coefficients import CoefficientManager
from .compute_theory import make_predictions
from .loader import load_datasets


class FitManager:
    """
    Class to collect all the fit information,
    load the results, compute best theory predictions.

    Attributes
    ----------
        path: pathlib.Path
            path to fit location
        name: srt
            fit name
        label: str, optional
            fit label if any otherwise guess it from the name
        config: dict
            configuration dictionary
        has_posterior: bool
            True if the fi contains the full posterrio distribution,
            False if only cl bounds are stored (external fits for benchmark)
        results: pandas.DataFrame
            fit results, they need to be loaded by `load_results`

    Parameters
    ----------
        path: pathlib.Path
            path to fit location
        name: srt
            fit name
        label: str, optional
            fit label if any otherwise guess it from the name
    """

    def __init__(self, path, name, label=None):
        self.path = path
        self.name = name
        self.label = (
            r"${\rm %s}$" % name.replace("_", r"\ ") if label is None else label
        )

        # load the configuration file
        self.config = self.load_configuration()
        self.has_posterior = self.config.get("has_posterior", True)
        self.results = None
        self.datasets = None
        self.rgemat = None

    def __repr__(self):
        return self.name

    def __eq__(self, comapre_name):
        return self.name == comapre_name

    def load_results(self):
        """
        Load posterior distribution of a fit.
        If the fit is produced by and external source it loads
        the results. Results are stored in a class attribute
        """
        file = "results"
        if self.has_posterior:
            file = "fit_results"
        with open(f"{self.path}/{self.name}/{file}.json", encoding="utf-8") as f:
            results = json.load(f)

        # load the rge matrix in the result dir if it exists
        try:
            with open(f"{self.path}/{self.name}/rge_matrix.pkl", "rb") as f:
                rgemats = pickle.load(f)
                self.operators_to_keep = {op: {} for op in rgemats[0].index}
                self.rgemat = jnp.stack([rgemat.values for rgemat in rgemats])

        except FileNotFoundError:
            print("No RGE matrix found in the result folder, skipping...")

        # if the posterior is from single parameter fits
        # then each distribution might have a different number of samples
        is_single_param = results.get("single_parameter_fits", False)
        if is_single_param:
            del results["single_parameter_fits"]

            num_samples = []
            for key in results["samples"].keys():
                num_samples.append(len(results["samples"][key]))
            num_samples_min = min(num_samples)

            for key in results["samples"].keys():
                results["samples"][key] = np.random.choice(
                    results["samples"][key], num_samples_min, replace=False
                )

        # TODO: support pariwise posteriors

        # Be sure columns are sorted, otherwise can't compute theory...
        results["samples"] = pd.DataFrame(results["samples"]).sort_index(axis=1)
        results["best_fit_point"] = pd.DataFrame(
            [results["best_fit_point"]]
        ).sort_index(axis=1)
        self.results = results

    def load_configuration(self):
        """Load configuration yaml card.

        Returns
        -------
        dict
            configuration card

        """
        with open(f"{self.path}/{self.name}/{self.name}.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    def load_datasets(self):
        """Load all datasets."""

        self.datasets = load_datasets(
            self.config["data_path"],
            self.config["datasets"],
            self.config["coefficients"]
            if self.rgemat is None
            else self.operators_to_keep,
            self.config["use_quad"],
            self.config["use_theory_covmat"],
            False,  # t0 is not used here because in the report we look at the experimental chi2
            self.config.get("use_multiplicative_prescription", False),
            self.config.get("default_order", "LO"),
            self.config.get("theory_path", None),
            self.config.get("rot_to_fit_basis", None),
            self.config.get("uv_couplings", False),
            self.config.get("external_chi2", False),
            rgemat=self.rgemat,
        )

    @property
    def smeft_predictions(self):
        """Compute |SMEFT| predictions for each replica.

        Returns
        -------
        np.ndarray:
            |SMEFT| predictions for each replica
        """

        smeft = []
        for rep in track(
            range(self.n_replica),
            description=f"[green]Computing SMEFT predictions for each replica of {self.name}...",
        ):
            smeft.append(
                make_predictions(
                    self.datasets,
                    self.results["samples"].iloc[rep, :],
                    self.config["use_quad"],
                    self.config.get("use_multiplicative_prescription", False),
                )
            )
        return np.array(smeft)

    @property
    def smeft_predictions_best_fit(self):
        """Compute |SMEFT| predictions for the best fit point.

        Returns
        -------
        np.ndarray:
            |SMEFT| predictions for the best fit
        """
        predictions = make_predictions(
            self.datasets,
            self.results["best_fit_point"].iloc[0, :],
            self.config["use_quad"],
            self.config.get("use_multiplicative_prescription", False),
        )

        # Add a dimension to match the shape of the replica predictions
        return np.array([predictions])

    @property
    def coefficients(self):
        """coefficient manager"""
        return CoefficientManager.from_dict(self.config["coefficients"])

    @property
    def n_replica(self):
        """Number of replicas"""
        return self.results["samples"].shape[0]
