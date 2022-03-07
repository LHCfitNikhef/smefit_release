# -*- coding: utf-8 -*-

"""
Fitting the Wilson coefficients with NS
"""
import os

import numpy as np

from .loader import aggregate_coefficients, load_datasets
from .optimize import Optimizer

# from mpi4py import MPI
# from pymultinest.solve import solve


class NSOptimizer(Optimizer):

    """Optimizer specification for |NS|"""

    def __init__(
        self,
        live_points,
        efficiency,
        const_efficiency,
        tollerance,
        loaded_datasets,
        coefficients,
        HOindex1,
        HOindex2,
    ):

        self.live_points = live_points
        self.efficiency = efficiency
        self.const_efficiency = const_efficiency
        self.tollerance = tollerance

        super().__init__(loaded_datasets, coefficients, HOindex1, HOindex2)

        # Get free parameters
        self.get_free_params()
        self.npar = len(self.free_params.keys())

    @classmethod
    def from_dict(cls, config):
        """
        Create object from theory dictionary.
        Parameters
        ----------
            config : dict
                config dictionary
        Returns
        -------
            cls : Optimizer
                created object
        """

        loaded_datasets = load_datasets(config["root_path"], config["datasets"])
        coefficients = aggregate_coefficients(config["coefficients"], loaded_datasets)

        for k in config["coefficients"]:
            if k not in coefficients.labels:
                raise NotImplementedError(
                    f"{k} does not enter the theory. Comment it out in setup script and restart."
                )
        # Get indice locations for quadratic corrections
        if config["HOlambda"] == "HO":
            HOindex1 = []
            HOindex2 = []

            for coeff in loaded_datasets.HOcorrectionsKEYS:
                idx1 = np.where(coefficients.labels == coeff.split("*")[0])[0][0]
                idx2 = np.where(coefficients.labels == coeff.split("*")[1])[0][0]
                HOindex1.append(idx1)
                HOindex2.append(idx2)
            HOindex1 = np.array(HOindex1)
            HOindex2 = np.array(HOindex2)
        else:
            HOindex1 = None
            HOindex2 = None

        if "nlive" in config.keys():
            live_points = config["nlive"]
        else:
            print(
                "Number of live points (nlive) not set in the input card. Using default: 500"
            )
            live_points = 500

        if "efr" in config.keys():
            efficiency = config["efr"]
        else:
            print(
                "Sampling efficiency (efr) not set in the input card. Using default: 0.01"
            )
            efficiency = 0.01

        if "ceff" in config.keys():
            const_efficiency = config["ceff"]
        else:
            print(
                "Constant efficiency mode (ceff) not set in the input card. Using default: False"
            )
            const_efficiency = False

        if "toll" in config.keys():
            tollerance = config["toll"]
        else:
            print(
                "Evidence tollerance (toll) not set in the input card. Using default: 0.5"
            )
            tollerance = 0.5

        return cls(
            live_points,
            efficiency,
            const_efficiency,
            tollerance,
            loaded_datasets,
            coefficients,
            HOindex1,
            HOindex2,
        )

    def chi2_func_ns(self, params):
        """
        Wrap the chi2 in a function for scipy optimiser. Pass noise and
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
        self.free_params = params
        self.propagate_params()
        # self.set_constraints()

        return self.chi2_func()

    def myloglike(self, hypercube):
        """
        Multi gaussian log likelihood function

        Parameters
        ----------
            hypercube :  np.ndarray
                hypercube prior

        Returns
        -------
            -0.5 * chi2 : np.ndarray
                multi gaussian log likelihood
        """

        return -0.5 * self.chi2_func_ns(hypercube)

    def myprior(self, hypercube):
        """
        Update the prior function

        Parameters
        ----------
            hypercube :  np.ndarray
                hypercube prior

        Returns
        -------
            hypercube : np.ndarray
                hypercube prior
        """

        for k, label in enumerate(self.free_params.keys()):

            idx = np.where(self.coefficients.labels == label)[0][0]
            min_val = self.coefficients.bounds[idx][0]
            max_val = self.coefficients.bounds[idx][1]
            hypercube[k] = hypercube[k] * (max_val - min_val) + min_val

        return hypercube

    # def clean(self):
    #     """Remove raw NS output if you want to keep raw output, don't call this method"""

    #     filelist = [
    #         f for f in os.listdir(self.config["results_path"]) if f.startswith("1k-")
    #     ]
    #     for f in filelist:
    #         if f in os.listdir(self.config["results_path"]):
    #             os.remove(os.path.join(self.config["results_path"], f))

    def run_sampling(self):
        """Run the minimisation with |NS|"""
        print("==================================")
        print("Run NS")
