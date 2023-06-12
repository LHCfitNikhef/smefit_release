# -*- coding: utf-8 -*-

"""
Fitting the Wilson coefficients with NS
"""
import time

from dynesty import NestedSampler
from dynesty.pool import Pool
from rich.style import Style
from rich.table import Table

from .. import log
from ..coefficients import CoefficientManager
from ..loader import load_datasets
from . import Optimizer

_logger = log.logging.getLogger(__name__)


class DynestyOptimizer(Optimizer):
    """
    Optimizer specification for Dynesty.

    Parameters
    ----------
        loaded_datasets : `smefit.loader.DataTuple`,
            dataset tuple
        coefficients : `smefit.coefficients.CoefficientManager`
            instance of `CoefficientManager` with all the relevant coefficients to fit
        use_quad : bool
            If True use also |HO| corrections
        nlive : int
            number of |NS| live points
        efficiency : float
            sampling efficiency
        const_efficiency: bool
            if True use the constant efficiency mode
        tolerance: float
            evidence tolerance
    """

    print_rate = 5000

    def __init__(
        self,
        loaded_datasets,
        coefficients,
        result_path,
        use_quad,
        result_ID,
        single_parameter_fits,
        pairwise_fits,
        use_multiplicative_prescription,
        nlive=500,
        n_pools=10,
    ):
        super().__init__(
            f"{result_path}/{result_ID}",
            loaded_datasets,
            coefficients,
            use_quad,
            single_parameter_fits,
            use_multiplicative_prescription,
        )
        self.nlive = nlive
        self.n_pools = n_pools
        self.npar = self.free_parameters.shape[0]
        self.result_ID = result_ID
        self.pairwise_fits = pairwise_fits

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
            config["use_t0"],
            config.get("use_multiplicative_prescription", False),
            config.get("theory_path", None),
            config.get("rot_to_fit_basis", None),
            config.get("uv_coupligs", False),
        )

        coefficients = CoefficientManager.from_dict(config["coefficients"])

        single_parameter_fits = config.get("single_parameter_fits", False)
        pairwise_fits = config.get("pairwise_fits", False)

        nlive = config.get("nlive", 500)
        if "nlive" not in config:
            _logger.warning(
                f"Number of live points (nlive) not set in the input card. Using default: {nlive}"
            )

        npools = config.get("npools", 8)
        if "npools" not in config:
            _logger.warning(
                f"Number of prallel pools not set in the input card. Using default: {npools}"
            )

        use_multiplicative_prescription = config.get(
            "use_multiplicative_prescription", False
        )
        return cls(
            loaded_datasets,
            coefficients,
            config["result_path"],
            config["use_quad"],
            config["result_ID"],
            single_parameter_fits,
            pairwise_fits,
            use_multiplicative_prescription,
            nlive=nlive,
            n_pools=npools,
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
        self.coefficients.set_free_parameters(params)
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

    def run_sampling(self):
        """Run the minimization with Dynesty Nested Sampling."""

        t1 = time.time()
        with Pool(
            self.n_pools,
            loglike=self.gaussian_loglikelihood,
            prior_transform=self.flat_prior,
        ) as pool:
            sampler = NestedSampler(
                pool.loglike,
                pool.prior_transform,
                self.npar,
                nlive=self.nlive,
                pool=pool,
            )
            sampler.run_nested()
            results = sampler.results
        t2 = time.time()

        log.console.log(f"Time : {((t2 - t1) / 60.0):.3f} minutes")
        log.console.log(f"Number of samples: {results['samples'].shape[0]}")
        table = Table(style=Style(color="white"), title_style="bold cyan", title=None)
        table.add_column("Parameter", style="bold red", no_wrap=True)
        table.add_column("Best value")
        table.add_column("Error")
        for par, col in zip(self.free_parameters.index, results["samples"].T):
            table.add_row(f"{par}", f"{col.mean():.3f}", f"{col.std():.3f}")
        log.console.print(table)
        self.save(results)

    def save(self, result):
        """
        Save |NS| replicas to json inside a dictionary:
        {coff: [replicas values]}

        Parameters
        ----------
            result : dict
                result dictionary

        """
        values = {}
        for c in self.coefficients.name:
            values[c] = []

        for sample in result["samples"]:

            self.coefficients.set_free_parameters(sample)
            self.coefficients.set_constraints()

            for c in self.coefficients.name:
                values[c].append(self.coefficients[c].value)

        if self.pairwise_fits:
            posterior_file = (
                self.results_path
                / f"posterior_{self.coefficients.name[0]}_{self.coefficients.name[1]}.json"
            )
        else:
            posterior_file = self.results_path / "posterior.json"

        self.dump_posterior(posterior_file, values)
