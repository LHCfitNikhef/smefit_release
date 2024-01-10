# -*- coding: utf-8 -*-
import pathlib

import numpy as np
import pandas as pd
import yaml

from ..compute_theory import make_predictions
from ..covmat import covmat_from_systematics
from ..loader import Loader, load_datasets
from ..log import logging

_logger = logging.getLogger(__name__)


class Projection:
    def __init__(
        self,
        commondata_path,
        theory_path,
        dataset_names,
        projections_path,
        coefficients,
        order,
        use_quad,
        rot_to_fit_basis,
    ):

        self.commondata_path = commondata_path
        self.theory_path = theory_path
        self.dataset_names = dataset_names
        self.projections_path = projections_path
        self.coefficients = coefficients
        self.order = order
        self.use_quad = use_quad
        self.rot_to_fit_basis = rot_to_fit_basis

        self.datasets = load_datasets(
            self.commondata_path,
            self.dataset_names,
            self.coefficients,
            self.order,
            self.use_quad,
            False,
            False,
            False,
            theory_path=self.theory_path,
        )

        if self.coefficients:
            _logger.info(
                f"Some coefficients are specified in the runcard: EFT correction will be used for the central values"
            )

    @classmethod
    def from_config(cls, projection_card):
        with open(projection_card, encoding="utf-8") as f:
            projection_config = yaml.safe_load(f)

        commondata_path = pathlib.Path(projection_config["commondata_path"]).absolute()
        theory_path = pathlib.Path(projection_config["theory_path"]).absolute()
        projections_path = pathlib.Path(
            projection_config["projections_path"]
        ).absolute()
        dataset_names = projection_config["datasets"]

        coefficients = projection_config.get("coefficients", [])
        order = projection_config.get("order", "LO")
        use_quad = projection_config.get("use_quad", False)
        rot_to_fit_basis = projection_config.get("rot_to_fit_basis", None)

        return cls(
            commondata_path,
            theory_path,
            dataset_names,
            projections_path,
            coefficients,
            order,
            use_quad,
            rot_to_fit_basis,
        )

    def compute_cv_projection(self):
        """
        Computes the new central value under the EFT hypothesis (is SM when coefficients are zero)

        Returns
        -------
            cv : numpy.ndarray
                SM + EFT theory predictions
        """
        cv = self.datasets.SMTheory

        if self.coefficients:
            coefficient_values = []
            for coeff in self.datasets.OperatorsNames:
                coefficient_values.append(self.coefficients[coeff]["value"])
            cv = make_predictions(
                self.datasets, coefficient_values, self.use_quad, False
            )
        return cv

    def build_projection(self, lumi_new):
        """
        Constructs runcard for projection by updating the central value and statistical uncertainties

        Parameters
        ----------
        lumi_new: float
            Adjusts the statistical uncertainties according to the specified luminosity
        """

        # compute central values under projection
        cv = self.compute_cv_projection()

        cnt = 0
        for dataset_idx, ndat in enumerate(self.datasets.NdataExp):

            dataset_name = self.datasets.ExpNames[dataset_idx]
            path_to_dataset = self.commondata_path / f"{dataset_name}.yaml"

            _logger.info(f"Building projection for : {dataset_name}")

            with open(path_to_dataset, encoding="utf-8") as f:
                data_dict = yaml.safe_load(f)

            idxs = slice(cnt, cnt + ndat)

            # statistical uncertainties get reduced by lumi_old/lumi_new
            lumi_old = self.datasets.Luminosity[dataset_idx]
            reduction_factor = lumi_old / lumi_new
            # replace stat with rescaled ones
            stat = np.asarray(data_dict["statistical_error"])

            # skip dataset when no separation between systematics and statistical uncertainties are provided
            if not np.any(stat):
                _logger.warning(
                    f"No separation between stat and sys uncertainties provided, skipping: {dataset_name}"
                )
                cnt += ndat
                continue

            data_dict["statistical_error"] = (reduction_factor * stat).tolist()

            # get systematics
            if isinstance(data_dict["sys_names"], str):
                sys = pd.DataFrame(data_dict["systematics"], [data_dict["sys_names"]]).T
            else:
                sys = pd.DataFrame(data_dict["systematics"], data_dict["sys_names"]).T

            # build covmat for projections. Use rescaled stat
            newcov = covmat_from_systematics([reduction_factor * stat], [sys])
            # add L1 noise to cv
            cv_projection = np.random.multivariate_normal(cv[idxs], newcov)

            # replace cv with updated central values
            if len(cv_projection) > 1:
                data_dict["data_central"] = cv_projection.tolist()
            else:
                data_dict["data_central"] = float(cv_projection[0])

            projection_folder = self.projections_path
            projection_folder.mkdir(exist_ok=True)
            with open(f"{projection_folder}/{dataset_name}.yaml", "w") as file:
                yaml.dump(data_dict, file, sort_keys=False)

            cnt += ndat
