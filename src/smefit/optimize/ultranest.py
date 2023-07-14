# -*- coding: utf-8 -*-

"""
Fitting the Wilson coefficients with |NS|
"""
import time

import ultranest
from rich.style import Style
from rich.table import Table
from ultranest import stepsampler

from .. import log
from ..coefficients import CoefficientManager
from ..loader import load_datasets
from . import Optimizer

try:
    from mpi4py import MPI

    run_parallel = True
except ModuleNotFoundError:
    run_parallel = False

_logger = log.logging.getLogger(__name__)


class USOptimizer(Optimizer):
    """
    Optimizer specification for Ultra nest

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
        lepsilon : float
            sampling tollerance. Terminate when live point likelihoods are all the same, within Lepsilon tolerance.
            Increase this when your likelihood function is inaccurate,
        target_evidence_unc: float
            target evidence uncertainty.
        target_post_unc: float
            target posterior uncertainty.
        frac_remain: float
            integrate until this fraction of the integral is left in the remainder.
            Set to a higher number (0.5) if you know the posterior is simple.
        store_raw: bool
            if True store the result to eventually resume the job
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
        live_points=500,
        lepsilon=0.001,
        target_evidence_unc=0.5,
        target_post_unc=0.5,
        frac_remain=0.01,
        store_raw=False,
    ):
        super().__init__(
            f"{result_path}/{result_ID}",
            loaded_datasets,
            coefficients,
            use_quad,
            single_parameter_fits,
            use_multiplicative_prescription,
        )
        self.live_points = live_points
        self.lepsilon = lepsilon
        self.target_evidence_unc = target_evidence_unc
        self.target_post_unc = target_post_unc
        self.frac_remain = frac_remain
        self.npar = self.free_parameters.shape[0]
        self.result_ID = result_ID
        self.pairwise_fits = pairwise_fits
        self.store_raw = store_raw

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

        lepsilon = config.get("lepsilon", 0.05)
        if "lepsilon" not in config:
            _logger.warning(
                f"Sampling tollerance (Lepsilon) not set in the input card. Using default: {lepsilon}"
            )

        target_evidence_unc = config.get("target_evidence_unc", 0.5)
        if "target_evidence_unc" not in config:
            _logger.warning(
                f"Target Evidence uncertanty (target_evidence_unc) not set in the input card. Using default: {target_evidence_unc}"
            )

        target_post_unc = config.get("target_post_unc", 0.5)
        if "target_post_unc" not in config:
            _logger.warning(
                f"Target Posterior uncertanty (target_post_unc) not set in the input card. Using default: {target_post_unc}"
            )

        frac_remain = config.get("frac_remain", 0.01)
        if "frac_remain" not in config:
            _logger.warning(
                f"Remaining fraction (frac_remain) not set in the input card. Using default: {frac_remain}"
            )

        store_raw = config.get("store_raw", False)

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
            live_points=nlive,
            lepsilon=lepsilon,
            target_evidence_unc=target_evidence_unc,
            target_post_unc=target_post_unc,
            frac_remain=frac_remain,
            store_raw=store_raw,
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
        min_val = self.free_parameters.minimum.values
        max_val = self.free_parameters.maximum.values
        return hypercube * (max_val - min_val) + min_val

    def run_sampling(self):
        """Run the minimization with Ultra nest."""

        log_dir = None
        if self.store_raw:
            log_dir = self.results_path

        t1 = time.time()
        sampler = ultranest.ReactiveNestedSampler(
            self.free_parameters.index.tolist(),
            self.gaussian_loglikelihood,
            self.flat_prior,
            log_dir=log_dir,
            resume=True,
        )
        if self.npar > 10:
            # set up step sampler. Here, we use a differential evolution slice sampler:
            sampler.stepsampler = stepsampler.SliceSampler(
                nsteps=100,
                generate_direction=stepsampler.generate_region_oriented_direction,
            )
        result = sampler.run(
            min_num_live_points=self.live_points,
            dlogz=self.target_evidence_unc,
            frac_remain=self.frac_remain,
            dKL=self.target_post_unc,
            Lepsilon=self.lepsilon,
            update_interval_volume_fraction=0.8 if self.npar > 20 else 0.2,
            max_num_improvement_loops=0,
        )

        t2 = time.time()
        rank = 0
        if run_parallel:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

        if rank == 0:
            log.console.log(f"Time : {((t2 - t1) / 60.0):.3f} minutes")
            log.console.log(f"Number of samples: {result['samples'].shape[0]}")

            table = Table(
                style=Style(color="white"), title_style="bold cyan", title=None
            )
            table.add_column("Parameter", style="bold red", no_wrap=True)
            table.add_column("Best value")
            table.add_column("Error")
            for par, col in zip(self.free_parameters.index, result["samples"].T):
                table.add_row(f"{par}", f"{col.mean():.3f}", f"{col.std():.3f}")
            log.console.print(table)

            self.save(result)

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
