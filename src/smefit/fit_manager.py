# -*- coding: utf-8 -*-
import json

import pandas as pd
import numpy as np
import yaml


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

        self.has_posterior = (
            self.config["has_posterior"] if "has_posterior" in self.config else True
        )

        self.results = None

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
            file = "posterior"
        with open(f"{self.path}/{self.name}/{file}.json", encoding="utf-8") as f:
            results = json.load(f)

        # if the posterior is from single parameter fits
        # then each distribution might have a different number of samples

        if results["single_parameter_fits"]:

            del results["single_parameter_fits"]

            num_samples = []
            for key in results.keys():
                num_samples.append(len(results[key]))
            num_samples_min = min(num_samples)

            for key in results.keys():
                results[key] = np.random.choice(
                    results[key], num_samples_min, replace=False
                )

        self.results = pd.DataFrame(results)

    def load_configuration(self):
        """
        Load configuration yaml card

        Returns
        -------
            config: dict
                configuration card
        """
        with open(f"{self.path}/{self.name}/{self.name}.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    @property
    def Nrep(self):
        """Number of replicas"""
        return self.results.shape[0]
