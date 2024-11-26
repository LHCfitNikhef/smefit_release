# -*- coding: utf-8 -*-
"""Solve the linear plolem to get the best analytic bounds."""
import numpy as np
from rich.style import Style
from rich.table import Table

from .. import chi2, log
from ..analyze.pca import impose_constrain
from ..coefficients import CoefficientManager
from ..loader import load_datasets
from . import Optimizer

_logger = log.logging.getLogger(__name__)


def is_semi_pos_def(x):
    """Check is a matrix is positive-semidefinite."""
    return np.all(np.linalg.eigvals(x) >= 0)


class ALOptimizer(Optimizer):
    """Optimizer specification for the linear analytic solution.

    Parameters
    ----------
    loaded_datasets : `smefit.loader.DataTuple`,
        dataset tuple
    coefficients : `smefit.coefficients.CoefficientManager`
        instance of `CoefficientManager` with all the relevant coefficients to fit
    result_path: pathlib.Path
        path to result folder
    result_ID : str
        result name
    single_parameter_fits : bool
        True for individual scan fits
    n_samples:
        number of replica to sample
    """

    def __init__(
        self,
        loaded_datasets,
        coefficients,
        result_path,
        result_ID,
        single_parameter_fits,
        n_samples,
    ):
        super().__init__(
            results_path=f"{result_path}/{result_ID}",
            loaded_datasets=loaded_datasets,
            coefficients=coefficients,
            # disble quadratic corrections here
            use_quad=False,
            single_parameter_fits=single_parameter_fits,
            # this option does not make any difference here
            use_multiplicative_prescription=False,
        )
        self.n_samples = n_samples

    @classmethod
    def from_dict(cls, config):
        """Create object from theory dictionary.

        Parameters
        ----------
        config : dict
            configuration dictionary

        Returns
        -------
        cls : Optimizer
            created object
        """
        use_quad = config["use_quad"]
        if use_quad:
            raise ValueError(
                "Analytic solution with quadratic corrections is not available."
            )

        loaded_datasets = load_datasets(
            config["data_path"],
            config["datasets"],
            config["coefficients"],
            False,
            config["use_theory_covmat"],
            config["use_t0"],
            False,
            config.get("theory_path", None),
            config.get("rot_to_fit_basis", None),
            config.get("uv_couplings", False),
        )

        coefficients = CoefficientManager.from_dict(config["coefficients"])
        single_parameter_fits = config.get("single_parameter_fits", False)
        return cls(
            loaded_datasets,
            coefficients,
            config["result_path"],
            config["result_ID"],
            single_parameter_fits,
            config["n_samples"],
        )

    def log_result(self, coeff_best, coeff_covmat):
        """Log a table with solution."""
        table = Table(style=Style(color="white"), title_style="bold cyan", title=None)
        table.add_column("Parameter", style="bold red", no_wrap=True)
        table.add_column("Best value")
        table.add_column("Error")
        for par, val, var in zip(
            self.free_parameters.index, coeff_best, np.diag(coeff_covmat)
        ):
            table.add_row(f"{par}", f"{val:.3f}", f"{np.sqrt(var):.3f}")
        log.console.print(table)

    def run_sampling(self):
        """Run sapmling accordying to the analytic solution."""

        fit_result = {}

        # update linear corrections in casde
        new_LinearCorrections = impose_constrain(
            self.loaded_datasets, self.coefficients
        )

        # compute mean and cov
        _logger.info("Computing Analytic solution ...")
        fisher = (
            new_LinearCorrections
            @ self.loaded_datasets.InvCovMat
            @ new_LinearCorrections.T
        )
        diff_sm = self.loaded_datasets.Commondata - self.loaded_datasets.SMTheory
        coeff_covmat = np.linalg.inv(fisher)

        # check if there are not flat directions
        if not is_semi_pos_def(coeff_covmat):
            raise ValueError(
                """Coefficient covariance is not symmetric positive-semidefinite,
                There might be flat directions to comment out."""
            )

        coeff_best = (
            coeff_covmat
            @ new_LinearCorrections
            @ self.loaded_datasets.InvCovMat
            @ diff_sm
        )

        # Compute some metrics of the fit result
        # Get names of coefficients, including the constrained ones
        coeffs_name = sorted(self.coefficients.name)

        # set the best fit point
        self.coefficients.set_free_parameters(coeff_best)
        self.coefficients.set_constraints()

        fit_result["best_fit_point"] = dict(zip(coeffs_name, self.coefficients.value))

        # compute max log likelihood
        chi2_tot = chi2.compute_chi2(
            self.loaded_datasets,
            self.coefficients.value,
            self.use_quad,
            self.use_multiplicative_prescription,
        )
        max_logl = float(-0.5 * chi2_tot)

        fit_result["max_loglikelihood"] = max_logl

        gaussian_integral = np.log(np.sqrt(np.linalg.det(2 * np.pi * coeff_covmat)))
        # NOTE: current formula for logz is not inluding the prior penalty
        logz = gaussian_integral + max_logl
        fit_result["logz"] = logz
        _logger.warning("The logZ computation is not including the prior penalty.")

        # generate samples in case n_samples > 0
        if self.n_samples > 0:
            self.log_result(coeff_best, coeff_covmat)

            # sample
            _logger.info("Sampling solutions ...")
            fit_result["samples"] = np.random.multivariate_normal(
                coeff_best, coeff_covmat, size=(self.n_samples,)
            )
            self.save(fit_result)
        else:  # record only chi2 if no samples are requested
            self.coefficients.set_free_parameters(coeff_best)
            self.coefficients.set_constraints()

            chi2_tot = chi2.compute_chi2(
                self.loaded_datasets,
                self.coefficients.value,
                self.use_quad,
                self.use_multiplicative_prescription,
            )
            chi2_red = chi2_tot / self.loaded_datasets.Commondata.size

            with open(self.results_path / "chi2.dat", "a") as f:
                f.write(f"{chi2_red} \n")

    def save(self, result):
        """Save samples to json inside a dictionary: {coff: [replicas values]}.
        Saving also some basic information about the fit.

        Parameters
        ----------
        samples : np.array
            raw samples with shape (n_samples, n_free_param)

        """
        posterior_samples = {}
        for c in self.coefficients.name:
            posterior_samples[c] = []

        # propagate constrain
        for sample in result["samples"]:
            self.coefficients.set_free_parameters(sample)
            self.coefficients.set_constraints()

            for c in self.coefficients.name:
                posterior_samples[c].append(self.coefficients[c].value)

        result["samples"] = posterior_samples
        # save fit result
        self.dump_fit_result(self.results_path / "fit_results.json", result)
