# -*- coding: utf-8 -*-

"""
Fitting the Wilson coefficients with NS
"""
import time

import numpy as np
from pymultinest.solve import solve

from ..coefficients import CoefficientManager
from ..loader import load_datasets
from . import Optimizer


class NSOptimizer(Optimizer):

    """
    Optimizer specification for |NS|

    Parameters
    ----------
        loaded_datasets : `smefit.loader.DataTuple`,
            dataset tuple
        coefficients : `smefit.coefficients.CoefficientManager`
            instance of `CoefficientManager` with all the relevant coefficients to fit
        use_quad : bool
            If True use also |HO| corrections
        live_points : int
            number of |NS| live points
        efficiency : float
            sampling efficiency
        const_efficiency: bool
            if True use the constant efficiency mode
        tolerance: float
            evidence tolerance
    """

    def __init__(
        self,
        loaded_datasets,
        coefficients,
        result_path,
        use_quad,
        live_points=500,
        efficiency=0.01,
        const_efficiency=False,
        tolerance=0.5,
    ):
        super().__init__(result_path, loaded_datasets, coefficients, use_quad)
        self.live_points = live_points
        self.efficiency = efficiency
        self.const_efficiency = const_efficiency
        self.tolerance = tolerance
        self.npar = self.free_parameters.size

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
            config["use_quad"],
            config["theory_path"] if "theory_path" in config else None,
        )
        coefficients = CoefficientManager(config["coefficients"])

        for k in config["coefficients"]:
            if k not in coefficients.labels:
                raise NotImplementedError(
                    f"{k} does not enter the theory. Comment it out in setup script and restart."
                )
        if "nlive" not in config:
            print(
                "Number of live points (nlive) not set in the input card. Using default: 500"
            )

        if "efr" not in config:
            print(
                "Sampling efficiency (efr) not set in the input card. Using default: 0.01"
            )

        if "ceff" not in config:
            print(
                "Constant efficiency mode (ceff) not set in the input card. Using default: False"
            )

        if "toll" not in config:
            print(
                "Evidence tolerance (toll) not set in the input card. Using default: 0.5"
            )

        return cls(
            loaded_datasets,
            coefficients,
            config["result_path"],
            config["use_quad"],
            **config,
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
        self.free_parameters.value = params
        self.coefficients.set_constraints()

        return self.chi2_func()

    def gaussian_loglikelihood(self, hypercube):
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

    def flat_prior(self, hypercube):
        """
        Update the prior function

        Parameters
        ----------
            hypercube : np.ndarray
                hypercube prior

        Returns
        -------
            flat_prior : np.ndarray
                updated hypercube prior
        """

        min_val = self.free_parameters.min
        max_val = self.free_parameters.max
        return hypercube * (max_val - min_val) + min_val

    # def clean(self):
    #     """Remove raw NS output if you want to keep raw output, don't call this method"""

    #     filelist = [
    #         f for f in os.listdir(self.config["results_path"]) if f.startswith("1k-")
    #     ]
    #     for f in filelist:
    #         if f in os.listdir(self.config["results_path"]):
    #             os.remove(os.path.join(self.config["results_path"], f))

    def run_sampling(self):
        """Run the minimization with Nested Sampling"""

        # Prefix for results
        prefix = self.results_path / f"{self.live_points}_"

        # Additional check
        # Multinest will crash if the length of the results
        # path+post_equal_weights.txt is longer than 100.
        # you can solve this making you path or fit id shorter.
        # Otherwise you can hack /pymultinest/solve.py
        if len(f"{prefix}post_equal_weights.txt") >= 100:
            raise UserWarning(
                f"Py multinest support a buffer or maximum 100 characters: \
                    {prefix}post_equal_weights.txt is too long, \
                         please chose a shorter path or Fit ID"
            )

        t1 = time.time()

        result = solve(
            LogLikelihood=self.gaussian_loglikelihood,
            Prior=self.flat_prior,
            n_dims=self.npar,
            n_params=self.npar,
            outputfiles_basename=prefix,
            n_live_points=self.live_points,
            sampling_efficiency=self.efficiency,
            verbose=True,
            importance_nested_sampling=True,
            const_efficiency_mode=self.const_efficiency,
            evidence_tolerance=self.tolerance,
        )

        t2 = time.time()

        print("Time = ", (t2 - t1) / 60.0)
        print()
        print(f"evidence: %(logZ){result[0]:1f} +- %(logZerr){result[1]:1f}")
        print()
        print("parameter values:")
        for par, col in zip(self.free_param_labels, result["samples"].transpose()):
            print("%15s : %.3f +- %.3f" % (par, col.mean(), col.std()))
        print(len(result["samples"]))

        self.save(result)
        self.clean()
