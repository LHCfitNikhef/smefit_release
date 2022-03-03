# -*- coding: utf-8 -*-

"""
Fitting the Wilson coefficients with NS
"""
import os
import time

from .optimize import Optimizer

# from mpi4py import MPI
# from pymultinest.solve import solve


class NSOptimizer(Optimizer):

    """Optimizer specification for |NS|"""

    def __init__(self, config):

        super().__init__(config)

        # Get free parameters
        self.get_free_params()
        self.npar = len(self.free_params)

        # TODO same as parent class, here we need a class method
        print("============================")

        if "nlive" in self.config.keys():
            self.live_points = self.config["nlive"]
        else:
            print(
                "Number of live points (nlive) not set in the input card. Using default: 500"
            )
            self.live_points = 500

        if "efr" in self.config.keys():
            self.efficiency = self.config["efr"]
        else:
            print(
                "Sampling efficiency (efr) not set in the input card. Using default: 0.01"
            )
            self.efficiency = 0.01

        if "ceff" in self.config.keys():
            self.const_efficiency = self.config["ceff"]
        else:
            print(
                "Constant efficiency mode (ceff) not set in the input card. Using default: False"
            )
            self.const_efficiency = False

        if "toll" in self.config.keys():
            self.tollerance = self.config["toll"]
        else:
            print(
                "Evidence tollerance (toll) not set in the input card. Using default: 0.5"
            )
            self.tollerance = 0.5

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
        self.set_constraints()

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

        for i in range(self.npar):

            label = self.free_param_labels[i]

            if self.config["bounds"] is None:
                min_val = self.config["coefficients"][label]["min"]
                max_val = self.config["coefficients"][label]["max"]
            else:
                min_val, max_val = self.coefficients.bounds[label]

            hypercube[i] = hypercube[i] * (max_val - min_val) + min_val

        return hypercube

    def clean(self):
        """Remove raw NS output if you want to keep raw output, don't call this method"""

        filelist = [
            f for f in os.listdir(self.config["results_path"]) if f.startswith("1k-")
        ]
        for f in filelist:
            if f in os.listdir(self.config["results_path"]):
                os.remove(os.path.join(self.config["results_path"], f))

    def run_sampling(self):
        """Run the minimisation with |NS|"""
        print("==================================")
        print("Run NS")
