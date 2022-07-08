# -*- coding: utf-8 -*-

"""
Fitting the Wilson coefficients with MC
"""
import copy
import json

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
        self.npar = self.free_parameters.size
        self.result_ID = result_ID
        self.chi2_tr_values = []
        self.chi2_val_values = []
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

    def get_status(self, chi2_tr, chi2_val):

        if len(self.chi2_tr_values) == 0:
            self.chi2_tr_values.append(chi2_tr)

        if chi2_tr < self.chi2_tr_values[-1]:
            self.chi2_tr_values.append(chi2_tr)
            self.chi2_val_values.append(chi2_val)
            self.coeff_steps.append(self.free_parameters.value)
            self.epoch += 1

    def chi2_func_mc(self, params):
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
        current_chi2_tr, current_chi2_val = self.chi2_func(True)
        self.get_status(current_chi2_tr, current_chi2_val)

        return current_chi2_tr

    def run_sampling(self, use_lookback=True):
        """Run the minimization with Nested Sampling"""

        bounds = [
            (self.free_parameters.minimum[i], self.free_parameters.maximum[i])
            for i in range(0, self.free_parameters.value.size)
        ]
        scipy_min = opt.minimize(
            self.chi2_func_mc,
            self.free_parameters.value,
            method="trust-constr",
            bounds=bounds,
        )

        if use_lookback:
            # find minimum of chi2_val and its position
            min_chi2_val = min(self.chi2_val_values)
            index = self.chi2_val_values.index(min_chi2_val)
            # select corresponding coeffs values
            best_values = self.coeff_steps[index]
        else:
            best_values = np.array(scipy_min.x)

        return best_values

    def save(self, result):
        """
        Save MC replicas to json inside a dictionary:
        {coff: replica values}

        Parameters
        ----------
            result : dict
                result dictionary

        """
        # TODO: move to __init__?
        values = {}
        for c, value in zip(self.coefficients.op_name, result):
            values[c] = value

        with open(
            self.results_path
            / f"replica_{self.replica}/coefficients_rep_{self.replica}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(values, f)
