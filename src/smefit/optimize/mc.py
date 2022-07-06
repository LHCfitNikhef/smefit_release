# -*- coding: utf-8 -*-

"""
Fitting the Wilson coefficients with MC
"""
import json
import os
import time
import warnings
import copy
import numpy as np
import scipy.optimize as opt


from mpi4py import MPI
from pymultinest.solve import solve

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
    ):
        super().__init__(
            f"{result_path}/{result_ID}", loaded_datasets, coefficients, use_quad
        )
        self.npar = self.free_parameters.size
        self.result_ID = result_ID
        self.cost_values = []
        self.coeff_steps = []
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
        )


    def get_status(self, chi2):

        if len(self.cost_values) == 0:
            self.cost_values.append(chi2)
        
        if chi2 < self.cost_values[-1]:
            self.cost_values.append(chi2)
            self.coeff_steps.append(copy.deepcopy(self.free_parameters.value))
            print(chi2)
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
        current_chi2 = self.chi2_func(True)[0]
        self.get_status(current_chi2)

        return current_chi2


    def run_sampling(self):
        """Run the minimization with Nested Sampling"""

        print("running...")
        bounds = [(self.free_parameters.minimum[i], self.free_parameters.maximum[i]) for i in range(0,self.free_parameters.value.size)]
        scipy_min = opt.minimize(
            self.chi2_func_mc, self.free_parameters.value, method="trust-constr", bounds=bounds
        )
        # t1 = time.time()

        
        # t2 = time.time()

        # print("Time = ", (t2 - t1) / 60.0, "\n")
        # print(f"evidence: {result['logZ']:1f} +- {result['logZerr']:1f} \n")
        # print("parameter values:")
        # for par, col in zip(self.free_parameters, result["samples"].T):
        #     print(f"{par.op_name} : {col.mean():3f} +- {col.std():3f}")
        # print(f"Number of samples: {result['samples'].shape[0]}")


        # if rank == 0:
        #     self.save(result)
        #     self.clean()

    def save(self, result):
        """
        Save |NS| replicas to json inside a dictionary:
        {coff: [replicas values]}

        Parameters
        ----------
            result : dict
                result dictionary

        """
        # TODO: move to __init__?
        values = {}
        for c in self.coefficients.op_name:
            values[c] = []

        for sample in result["samples"]:

            self.coefficients.free_parameters.value = sample
            self.coefficients.set_constraints()

            for c in self.coefficients.op_name:
                values[c].append(float(self.coefficients.get_from_name(c).value))

        with open(self.results_path / "posterior.json", "w", encoding="utf-8") as f:
            json.dump(values, f)
