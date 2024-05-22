# -*- coding: utf-8 -*-
import importlib
import json
import pathlib
import sys

from rich.style import Style
from rich.table import Table

from .. import chi2, log
from ..loader import get_dataset

try:
    from mpi4py import MPI

    run_parallel = True
except ModuleNotFoundError:
    run_parallel = False


class Optimizer:
    """
    Common interface for Chi2 profile, NS and MC and A optimizers.

    Parameters
    ----------
        results_path : pathlib.path
            path to result folder
        loaded_datasets : DataTuple,
            dataset tuple
        coefficients : `smefit.coefficients.CoefficientManager`
            instance of `CoefficientManager` with all the relevant coefficients to fit
        use_quad : bool
            if True includes also |HO| correction
        single_parameter_fits : bool
            True for single parameter fits
        use_multiplicative_prescription:
            if True uses the multiplicative prescription for the |EFT| corrections.
    """

    print_rate = 500

    def __init__(
        self,
        results_path,
        loaded_datasets,
        coefficients,
        use_quad,
        single_parameter_fits,
        use_multiplicative_prescription,
        external_likelihoods=None,
    ):
        self.results_path = pathlib.Path(results_path)
        self.loaded_datasets = loaded_datasets
        self.coefficients = coefficients
        self.use_quad = use_quad
        self.npts = (
            self.loaded_datasets.Commondata.size
            if self.loaded_datasets is not None
            else 0
        )
        self.single_parameter_fits = single_parameter_fits
        self.use_multiplicative_prescription = use_multiplicative_prescription
        self.counter = 0

        # load external likelihood modules as amortized objects (fast to evaluate)
        self.external_likelihoods = (
            self.load_external_likelihoods(external_likelihoods) if external_likelihoods else None
        )

    def load_external_likelihoods(self, external_likelihoods):
        """
        Loads the external chi2 modules

        Parameters
        ----------
        external_chi2: dict
            dict of external chi2s, with the name of the function object as key and the path to the external script
            as value

        Returns
        -------
        ext_chi2_modules: list
             List of external chi2 objects that can be evaluated by passing a coefficients instance
        """
        # dynamical import
        ext_likelihood_modules = []
        for external_likelihood in external_likelihoods:

            likelihood_type = external_likelihood['likelihood_type']
            likelihood_module = importlib.import_module(likelihood_type)
            my_likelihood_class = getattr(likelihood_module, likelihood_type)

            # initialise external likelihood object
            del external_likelihood['likelihood_type']
            del external_likelihood['path']
            likelihood_ext = my_likelihood_class(self.coefficients, **external_likelihood)

            ext_likelihood_modules.append(likelihood_ext.compute_neg_log_likelihood)
        import pdb; pdb.set_trace()
        return ext_chi2_modules

    @property
    def free_parameters(self):
        """Returns the free parameters entering fit"""
        return self.coefficients.free_parameters

    def generate_chi2_table(self, chi2_dict, chi2_tot):
        r"""Generate log :math:`\chi^2` table"""
        table = Table(style=Style(color="white"), title_style="bold cyan", title=None)
        table.add_column("Dataset", style="bold green", no_wrap=True)

        table.add_column("Chi^2 /N_dat")
        for name, val in chi2_dict.items():
            table.add_row(str(name), f"{val:.5}")
        table.add_row("Total", f"{(chi2_tot/self.npts):.5}")

        return table

    def chi2_func(self, use_replica=False, print_log=True):
        r"""
        Wrap the math:`\chi^2` in a function for the optimizer. Pass noise and
        data info as args. Log the math:`\chi^2` value and values of the coefficients.

        Returns
        -------
            current_chi2 : np.ndarray
                computed :math:`\chi^2`
        """
        rank = 0
        if run_parallel:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

        if rank == 0:
            self.counter += 1
            if print_log:
                print_log = (self.counter % self.print_rate) == 0
        else:
            print_log = False

        # only compute the internal chi2 when datasets are loaded
        if self.loaded_datasets is not None:
            chi2_tot = chi2.compute_chi2(
                self.loaded_datasets,
                self.coefficients.value,
                self.use_quad,
                self.use_multiplicative_prescription,
                use_replica,
            )
        else:
            chi2_tot = 0

        if self.external_likelihoods is not None:
            for external_likelihood in self.external_likelihoods:
                external_likelihood_i = external_likelihood(self.coefficients.value)
                external_likelihood_tot += external_likelihood_i

        if print_log:
            chi2_dict = {}
            for data_name in self.loaded_datasets.ExpNames:
                dataset = get_dataset(self.loaded_datasets, data_name)
                chi2_dict[data_name] = (
                    chi2.compute_chi2(
                        dataset,
                        self.coefficients.value,
                        self.use_quad,
                        self.use_multiplicative_prescription,
                        use_replica,
                    )
                    / dataset.NdataExp
                )
            log.console.print(self.generate_chi2_table(chi2_dict, chi2_tot))

        return chi2_tot

    def dump_posterior(self, posterior_file, values):
        if self.single_parameter_fits:
            if posterior_file.is_file():
                with open(posterior_file, encoding="utf-8") as f:
                    tmp = json.load(f)
                    values.update(tmp)
            else:
                values["single_parameter_fits"] = True

        with open(posterior_file, "w", encoding="utf-8") as f:
            json.dump(values, f)
