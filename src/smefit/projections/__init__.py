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
        fred_tot,
        fred_sys,
    ):

        self.commondata_path = commondata_path
        self.theory_path = theory_path
        self.dataset_names = dataset_names
        self.projections_path = projections_path
        self.coefficients = coefficients
        self.order = order
        self.use_quad = use_quad
        self.rot_to_fit_basis = rot_to_fit_basis
        self.fred_tot = fred_tot
        self.fred_sys = fred_sys

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

        fred_tot = projection_config.get("fred_tot", 1)
        fred_sys = projection_config.get("fred_sys", 1)

        return cls(
            commondata_path,
            theory_path,
            dataset_names,
            projections_path,
            coefficients,
            order,
            use_quad,
            rot_to_fit_basis,
            fred_tot,
            fred_sys
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

            # statistical uncertainties get reduced by sqrt(lumi_old/lumi_new)
            lumi_old = self.datasets.Luminosity[dataset_idx]
            fred_stat = np.sqrt(lumi_old / lumi_new)

            # load the statistical and systematic uncertainties
            stat = np.asarray(data_dict["statistical_error"])
            if not isinstance(data_dict["sys_names"], list):  # for a single systematic (replace by check on number of syst?)
                sys = pd.DataFrame(data_dict["systematics"], [data_dict["sys_names"]]).T
            else:
                sys = pd.DataFrame(data_dict["systematics"], data_dict["sys_names"]).T

            # if all stat unc are zero, we rescale the total error by 1/3 (compromise)
            if not np.any(stat):

                # check if systematics are diagonal
                is_sys_diag = np.count_nonzero(sys - np.diag(np.diagonal(sys))) == 0
                if not is_sys_diag:

                    # reconstruct covmat and keep only diagonal components
                    cov_tot = sys @ sys.T
                    sys_diag = np.sqrt(np.diagonal(cov_tot))

                    # rescale systematics
                    sys_rescaled = np.diag(sys_diag * self.fred_sys)
                    sys = pd.DataFrame(sys_rescaled, index=sys.index, columns=sys.columns)
                    
                else:
                    sys = sys * self.fred_sys


                cnt += ndat
                continue
            else:
                continue

            # check for the breakdown of various sources of systematic uncertainties, characterised by a non-square
            # systematic entry
            #
            # TODO: if the number of sources equals the number of datapoints, this test fails)



            if sys.shape[0] == sys.shape[1]:  # square
                if sys.shape[0] == 1: # a single uncorrelated systematic
                    sys *= self.fred_sys
                else:
                    import pdb; pdb.set_trace()




            data_dict["statistical_error"] = (reduction_factor * stat).tolist()




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
            with open(f"{projection_folder}/{dataset_name}_proj.yaml", "w") as file:
                yaml.dump(data_dict, file, sort_keys=False)

            cnt += ndat
