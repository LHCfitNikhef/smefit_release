# -*- coding: utf-8 -*-

"""
Fitting the Wilson coefficients with MC
"""
import copy
import json

import numpy as np
import scipy.optimize as opt

from ..coefficients import CoefficientManager
from ..loader import load_datasets
from . import Optimizer


class MCOptimizer(Optimizer):

    """
    Optimizer specification for MC

    Parameters
    ----------
        loaded_datasets : `smefit.loader.DataTuple`,
            dataset tuple
        coefficients : `smefit.coefficients.CoefficientManager`
            instance of `CoefficientManager` with all the relevant coefficients to fit
        use_quad : bool
            If True use also |HO| corrections
    """

    def __init__(
        self,
        loaded_datasets,
        coefficients,
        result_path,
        use_quad,
        result_ID,
        replica,
    ):
        super().__init__(
            f"{result_path}/{result_ID}", loaded_datasets, coefficients, use_quad
        )
        self.chi2_values = []
        self.coeff_steps = []
        self.replica = replica
        self.epoch = 0

    @classmethod
    def from_dict(cls, config):
        """
        Create object from theory dictionary.

        Parameters
        ----------
            config : dict
                configuration dictionary

        Returns
        -------
            cls : Optimizer
                created object
        """

        loaded_datasets = load_datasets(
            config["data_path"],
            config["datasets"],
            config["coefficients"],
            config["order"],
            config["use_quad"],
            config["use_theory_covmat"],
            config["theory_path"] if "theory_path" in config else None,
            config["rot_to_fit_basis"] if "rot_to_fit_basis" in config else None,
        )

        missing_operators = []
        for k in config["coefficients"]:
            if k not in loaded_datasets.OperatorsNames:
                missing_operators.append(k)
        if missing_operators:
            raise NotImplementedError(
                f"{missing_operators} not in the theory. Comment it out in setup script and restart."
            )
        coefficients = CoefficientManager.from_dict(config["coefficients"])

        return cls(
            loaded_datasets,
            coefficients,
            config["result_path"],
            config["use_quad"],
            config["result_ID"],
            config["replica"],
        )

    def get_status(self, chi2):

        if len(self.chi2_values) == 0:
            self.chi2_values.append(chi2)

        if chi2 < self.chi2_values[-1]:
            self.chi2_values.append(chi2)
            self.coeff_steps.append(self.free_parameters.value)
            self.epoch += 1

    def chi2_func_mc(self, params, print_log=True):
        """
        Wrap the chi2 in a function for the optimizer. Pass noise and
        data info as args. Log the chi2 value and values of the coefficients.

        Parameters
        ----------
            params : np.ndarray
                noise and data info
        Returns
        -------
            current_chi2 : np.ndarray
                chi2 function

        """
        self.free_parameters.value = params
        self.coefficients.set_constraints()
        current_chi2 = self.chi2_func(True, print_log)
        self.get_status(current_chi2)

        return current_chi2

    def chi2_scan(self):
        """Individual chi2 scan"""
        free_temp = copy.deepcopy(self.free_parameters)
        for coeff in free_temp:
            coeff.is_free = False
            coeff.value = 0.0

        def regularized_chi2_func(xs):
            roots = []
            for x in xs:
                coeff.value = x
                chi2 = self.chi2_func_mc(free_temp.value, print_log=False)
                roots.append(chi2 / self.npts - 2.0)
            return roots

        bounds = []
        for coeff in free_temp:
            coeff.is_free = True
            roots = opt.fsolve(regularized_chi2_func, [-50, 50], xtol=1e-6)
            bounds.append(roots)
            coeff.is_free = False
            print(f"Chi^2 scan terminated, found bounds for {coeff.op_name}: {roots}")
        return bounds

    def run_sampling(self):
        """Run the minimization with Nested Sampling"""

        bounds = self.chi2_scan()
        # TODO: other minimization options?
        opt.minimize(
            self.chi2_func_mc,
            self.free_parameters.value,
            method="trust-constr",
            bounds=bounds,
        )

    def save(self):
        """
        Save MC replicas to json inside a dictionary:
        {coff: replica values}

        Parameters
        ----------
            result : dict
                result dictionary

        """
        values = {}
        values["chi2"] = self.chi2_values[-1] / self.npts
        for c, value in zip(self.coefficients.op_name, self.coefficients.value):
            values[c] = value

        with open(
            self.results_path
            / f"replica_{self.replica}/coefficients_rep_{self.replica}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(values, f)
