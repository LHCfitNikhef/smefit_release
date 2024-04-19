# -*- coding: utf-8 -*-
import pathlib
import shutil

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
        use_theory_covmat,
        rot_to_fit_basis,
        fred_tot,
        fred_sys,
        use_t0,
    ):

        self.commondata_path = commondata_path
        self.theory_path = theory_path
        self.dataset_names = dataset_names
        self.projections_path = projections_path
        self.coefficients = coefficients
        self.order = order
        self.use_quad = use_quad
        self.use_theory_covmat = use_theory_covmat
        self.rot_to_fit_basis = rot_to_fit_basis
        self.fred_tot = fred_tot
        self.fred_sys = fred_sys
        self.use_t0 = use_t0

        self.datasets = load_datasets(
            self.commondata_path,
            self.dataset_names,
            self.coefficients,
            self.order,
            self.use_quad,
            self.use_theory_covmat,
            self.use_t0,
            False,
            theory_path=self.theory_path,
        )

        if self.coefficients:
            _logger.info(
                f"Some coefficients are specified in the runcard: EFT correction will be used for the central values"
            )

    @classmethod
    def from_config(cls, projection_card):
        """
        Returns the class Projection

        Parameters
        ----------
        projection_card: pathlib.Path
            path to projection runcard

        Returns
        -------
        Projection class
        """
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
        use_theory_covmat = projection_config.get("use_theory_covmat", True)
        rot_to_fit_basis = projection_config.get("rot_to_fit_basis", None)

        fred_tot = projection_config.get("fred_tot", 1)
        fred_sys = projection_config.get("fred_sys", 1)

        use_t0 = projection_config.get("use_t0", False)

        return cls(
            commondata_path,
            theory_path,
            dataset_names,
            projections_path,
            coefficients,
            order,
            use_quad,
            use_theory_covmat,
            rot_to_fit_basis,
            fred_tot,
            fred_sys,
            use_t0,
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

    def rescale_sys(self, sys, fred_sys):
        """
        Projects the systematic uncertainties by reducing them by fred_sys

        Parameters
        ----------
        sys: numpy.ndarray
            systematics
        fred: float
            Systematics reduction factor

        Returns
        -------
        Projected systematic uncertainties
        """

        # check whether the systematics are artificial (i.e. no breakdown of separate systematic sources),
        # characterised by a square matrix and negative entries
        is_square = sys.shape[0] == sys.shape[1]
        is_artificial = is_square & np.any(sys < 0)

        if is_artificial:

            # reconstruct covmat and keep only diagonal components
            cov_tot = sys @ sys.T
            sys_diag = np.sqrt(np.diagonal(cov_tot))

            # rescale systematics and make df
            sys_rescaled = np.diag(sys_diag * fred_sys)
            return pd.DataFrame(sys_rescaled, index=sys.index, columns=sys.columns)

        return sys * fred_sys

    @staticmethod
    def rescale_stat(stat, lumi_old, lumi_new):

        """
        Projects the statistical uncertainties from lumi_old to lumi_new

        Parameters
        ----------
        stat: numpy.ndarray
            old statistical uncertainties
        lumi_old: float
            Old luminosity
        lumi_new: float
            New luminosity

        Returns
        -------
        Updated statistical uncertainties after projection
        """
        fred_stat = np.sqrt(lumi_old / lumi_new)
        return stat * fred_stat

    def build_projection(self, lumi_new, closure):
        """
        Constructs runcard for projection by updating the central value and statistical and
        systematic uncertainties

        Parameters
        ----------
        lumi_new: float
            Adjusts the statistical uncertainties according to the specified luminosity
        closure: bool
            Set to true for a L1 closure test (no rescaling, only cv gets fluctuated according to
            original uncertainties)
        """

        # compute central values under projection
        cv = self.compute_cv_projection()

        cnt = 0
        for dataset_idx, num_data in enumerate(self.datasets.NdataExp):

            dataset_name = self.datasets.ExpNames[dataset_idx]
            path_to_dataset = self.commondata_path / f"{dataset_name}.yaml"

            _logger.info(f"Building projection for : {dataset_name}")

            try:
                with open(path_to_dataset, encoding="utf-8") as f:
                    data_dict = yaml.safe_load(f)
            except FileNotFoundError:
                _logger.info(f"Dataset {dataset_name} not in the database")
                continue

            idxs = slice(cnt, cnt + num_data)

            central_values = np.array(data_dict["data_central"])
            cv_theory = cv[idxs]

            # ratio SM to experimental central value
            ratio_sm_exp = cv_theory / central_values

            # set negative ratios to one
            ratio_sm_exp[ratio_sm_exp < 0] = 1

            # rescale the statistical uncertainty to the SM
            stat = np.asarray(data_dict["statistical_error"]) * np.sqrt(ratio_sm_exp)

            # load systematics
            num_sys = data_dict["num_sys"]
            sys_add = np.array(data_dict["systematics"])

            if num_sys != 0:
                type_sys = np.array(data_dict["sys_type"])
                name_sys = data_dict["sys_names"]

                # express systematics as percentage values of the central values
                sys_mult = sys_add / central_values * 1e2

                # Identify add and mult systematics
                # and replace the mult ones with corresponding value computed
                # from data central value. Required for implementation of t0 prescription
                indx_add = np.where(type_sys == "ADD")[0]
                indx_mult = np.where(type_sys == "MULT")[0]
                sys_t0 = np.zeros((num_sys, num_data))
                sys_t0[indx_add] = sys_add[indx_add].reshape(sys_t0[indx_add].shape)
                sys_t0[indx_mult] = (
                    sys_mult[indx_mult].reshape(sys_t0[indx_mult].shape)
                    * cv_theory
                    * 1e-2
                )

                # limit case with 1 sys
                if num_sys == 1:
                    name_sys = [name_sys]
            # limit case no sys
            else:
                name_sys = ["UNCORR"]
                sys_t0 = np.zeros((num_sys + 1, num_data))

            # these are absolute systematics
            sys = pd.DataFrame(data=sys_t0.T, columns=name_sys)
            n_sys = data_dict["num_sys"]

            th_covmat = self.datasets.ThCovMat[idxs, idxs]

            if not closure:
                # if all stats are zero, we only have access to the total error which we rescale by 1/3 (compromise)
                no_stats = not np.any(stat)
                if no_stats:
                    fred = self.fred_tot
                    sys_red = self.rescale_sys(sys, fred)
                    stat_red = stat
                # if separate stat and sys
                else:
                    fred = self.fred_sys
                    lumi_old = self.datasets.Luminosity[dataset_idx]
                    stat_red = self.rescale_stat(stat, lumi_old, lumi_new)
                    sys_red = self.rescale_sys(sys, fred)

                if num_data > 1:
                    data_dict["systematics"] = sys_red.T.values.tolist()
                else:
                    data_dict["systematics"] = sys_red.T.values.flatten().tolist()

                if data_dict["sys_type"] is not None:
                    data_dict["sys_type"] = ["ADD"] * n_sys if n_sys > 1 else "ADD"

                data_dict["statistical_error"] = stat_red.tolist()

                # build covmat for projections. Use rescaled uncertainties
                newcov = covmat_from_systematics([stat_red], [sys_red])
            else:  # closure test

                # we store absolute uncertainties and convert all multipicative uncertainties to additive ones

                if num_data > 1:
                    data_dict["systematics"] = sys.T.values.tolist()
                else:
                    data_dict["systematics"] = sys.T.values.flatten().tolist()

                if data_dict["sys_type"] is not None:
                    data_dict["sys_type"] = ["ADD"] * n_sys if n_sys > 1 else "ADD"

                data_dict["statistical_error"] = stat.tolist()

                newcov = covmat_from_systematics([stat], [sys])

            if self.use_theory_covmat:
                newcov += th_covmat

            # add L1 noise to cv
            cv_projection = np.random.multivariate_normal(cv[idxs], newcov)

            # replace cv with updated central values
            if len(cv_projection) > 1:
                data_dict["data_central"] = cv_projection.tolist()
            else:
                data_dict["data_central"] = float(cv_projection[0])

            projection_folder = self.projections_path
            projection_folder.mkdir(exist_ok=True)

            if projection_folder != self.commondata_path:
                if not closure:
                    with open(
                        f"{projection_folder}/{dataset_name}_proj.yaml", "w"
                    ) as file:
                        yaml.dump(data_dict, file, sort_keys=False)
                else:
                    with open(f"{projection_folder}/{dataset_name}.yaml", "w") as file:
                        yaml.dump(data_dict, file, sort_keys=False)
            else:
                print(
                    "Choose a different projection folder from commondata to avoid overwriting results"
                )
                sys.exit()

            # copy corresponding theory predictions with _proj appended to filename
            if not closure:
                shutil.copy(
                    self.theory_path / f"{dataset_name}.json",
                    self.theory_path / f"{dataset_name}_proj.json",
                )

            cnt += num_data
