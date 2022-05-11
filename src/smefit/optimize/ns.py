# -*- coding: utf-8 -*-

"""
Fitting the Wilson coefficients with NS
"""
import json
import os
import time
import warnings

from mpi4py import MPI
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
        result_ID,
        live_points=500,
        efficiency=0.01,
        const_efficiency=False,
        tolerance=0.5,
    ):
        super().__init__(
            f"{result_path}/{result_ID}", loaded_datasets, coefficients, use_quad
        )
        self.live_points = live_points
        self.efficiency = efficiency
        self.const_efficiency = const_efficiency
        self.tolerance = tolerance
        self.npar = self.free_parameters.size
        self.result_ID = result_ID

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
            config["rot_to_fit_basis"] if "rot_to_fit_basis" in config else None,
        )

        for k in config["coefficients"]:
            if k not in loaded_datasets.OperatorsNames:
                raise NotImplementedError(
                    f"{k} does not enter the theory. Comment it out in setup script and restart."
                )
        coefficients = CoefficientManager.from_dict(config["coefficients"])

        if "nlive" not in config:
            print(
                "Number of live points (nlive) not set in the input card. Using default: 500"
            )
            nlive = 500
        else:
            nlive = config["nlive"]

        if "efr" not in config:
            warnings.warn(
                "Sampling efficiency (efr) not set in the input card. Using default: 0.01"
            )
            efr = 0.01
        else:
            efr = config["efr"]

        if "ceff" not in config:
            warnings.warn(
                "Constant efficiency mode (ceff) not set in the input card. Using default: False"
            )
            ceff = False
        else:
            ceff = config["ceff"]

        if "toll" not in config:
            warnings.warn(
                "Evidence tolerance (toll) not set in the input card. Using default: 0.5"
            )
            toll = 0.5
        else:
            toll = config["toll"]

        return cls(
            loaded_datasets,
            coefficients,
            config["result_path"],
            config["use_quad"],
            config["result_ID"],
            live_points=nlive,
            efficiency=efr,
            const_efficiency=ceff,
            tolerance=toll,
        )

    def chi2_func_ns(self, params):
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
        min_val = self.free_parameters.minimum
        max_val = self.free_parameters.maximum
        return hypercube * (max_val - min_val) + min_val

    def clean(self):
        """Remove raw |NS| output if you want to keep raw output, don't call this method"""

        for f in os.listdir(self.results_path):
            if f.startswith(f"{self.live_points}_"):
                os.remove(self.results_path / f)

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
            outputfiles_basename=str(prefix),
            n_live_points=self.live_points,
            sampling_efficiency=self.efficiency,
            verbose=True,
            importance_nested_sampling=True,
            const_efficiency_mode=self.const_efficiency,
            evidence_tolerance=self.tolerance,
        )

        t2 = time.time()

        print("Time = ", (t2 - t1) / 60.0, "\n")
        print(f"evidence: {result['logZ']:1f} +- {result['logZerr']:1f} \n")
        print("parameter values:")
        for par, col in zip(self.free_parameters, result["samples"].T):
            print(f"{par.op_name} : {col.mean():3f} +- {col.std():3f}")
        print(f"Number of samples: {result['samples'].shape[0]}")

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            self.save(result)
            self.clean()

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
