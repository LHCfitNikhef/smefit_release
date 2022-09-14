# -*- coding: utf-8 -*-

"""
Fitting the Wilson coefficients with MC
"""
import copy
import json
import time

import numpy as np
import scipy.optimize as opt
from rich.style import Style
from rich.table import Table
from scipy.optimize import Bounds

from .. import log
from ..chi2 import compute_chi2
from ..coefficients import CoefficientManager
from ..loader import load_datasets
from . import Optimizer

_logger = log.logging.getLogger(__name__)


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
        single_parameter_fits,
        use_bounds,
        maxiter,
    ):
        super().__init__(
            f"{result_path}/{result_ID}", loaded_datasets, coefficients, use_quad
        )
        self.chi2_values = []
        self.coeff_steps = []
        self.replica = replica
        self.epoch = 0
        self.single_parameter_fits = single_parameter_fits
        self.use_bounds = use_bounds
        self.maxiter = maxiter

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
            config.get("theory_path", None),
            config.get("rot_to_fit_basis", None),
            config.get("uv_coupligs", False),
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

        use_bounds = config.get("use_bounds", True)
        if not use_bounds:
            log.console.log("Running minimization without initial bounds")

        maxiter = config.get("maxiter", 1e4)
        if "maxiter" not in config:
            _logger.warning(
                "Number of maximum iterations (maxiter) not set in the input card. Using default: 1e4"
            )

        return cls(
            loaded_datasets,
            coefficients,
            config["result_path"],
            config["use_quad"],
            config["result_ID"],
            config["replica"],
            config["single_parameter_fits"],
            use_bounds,
            maxiter,
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
        self.coefficients.set_free_parameters(params)
        self.coefficients.set_constraints()
        current_chi2 = self.chi2_func(True, print_log)
        self.get_status(current_chi2)

        return current_chi2

    def chi2_scan(self):
        r"""Individual :math:`\Chi^2` scan"""

        # set all the coefficients to 0 and fixed
        coefficient_temp = copy.deepcopy(self.coefficients)
        for coeff in coefficient_temp:
            coeff.is_free = False
            coeff.value = 0.0

        # chi2 - 2 == 0
        def regularized_chi2_func(xs):
            roots = []
            for x in xs:
                coeff.value = x
                coefficient_temp.set_constraints()
                chi2 = compute_chi2(
                    self.loaded_datasets,
                    coefficient_temp.value,
                    self.use_quad,
                    False,
                )
                roots.append(chi2 / self.npts - 2.0)
            return roots

        # fin the bound for each coefficient
        bounds = []
        for coeff in coefficient_temp:
            if coeff not in self.free_parameters:
                continue
            coeff.is_free = True
            roots = opt.fsolve(
                regularized_chi2_func, [-1000, 1000], xtol=1e-6, maxfev=400
            )
            bounds.append(roots)
            coeff.is_free = False
            coeff.value = 0.0
            coefficient_temp.set_constraints()
            log.console.log(f"Chi^2 scan : bounds for {coeff.name}: {roots}")
        return bounds

    def run_sampling(self):
        """Run the minimization with Nested Sampling"""

        t1 = time.time()
        bounds = None
        if self.use_bounds:
            bounds = Bounds(self.free_parameters.minimum, self.free_parameters.maximum)

        # TODO: other minimization options?
        opt.minimize(
            self.chi2_func_mc,
            self.free_parameters.value,
            method="trust-constr",
            bounds=bounds,
            options={"maxiter": self.maxiter},
        )
        t2 = time.time()
        log.console.log(f"Time : {((t2 - t1) / 60.0):.3f} minutes")

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
        table = Table(style=Style(color="white"), title_style="bold cyan", title=None)
        table.add_column("Parameter", style="bold red", no_wrap=True)
        table.add_column("Best value")
        for par, value in zip(self.coefficients.name, self.coefficients.value):
            table.add_row(f"{par}", f"{value:.3f}")
            values[par] = value
        log.console.print(table)

        posterior_file = (
            self.results_path
            / f"replica_{self.replica}/coefficients_rep_{self.replica}.json"
        )

        # if it s a single parameter fit check if the posterior file is already present and in case it is update it
        if self.single_parameter_fits:
            if posterior_file.is_file():
                with open(posterior_file, encoding="utf-8") as f:
                    tmp = json.load(f)
                    values.update(tmp)
                    values["chi2"] = None

        with open(
            posterior_file,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(values, f)
