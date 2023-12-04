# -*- coding: utf-8 -*-
import json
import pathlib
import importlib
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
    Common interface for Chi2 profile, NS and McFiT

    Parameters
    ----------
        results_path : pathlib.path

        loaded_datasets : DataTuple,
            dataset tuple
        coefficients :

        use_quad :
            if True include also |HO| correction

    """

    print_rate = 500

    # TODO: docstring

    def __init__(
        self,
        results_path,
        loaded_datasets,
        coefficients,
        use_quad,
        single_parameter_fits,
        use_multiplicative_prescription,
        external_chi2,
    ):
        self.results_path = pathlib.Path(results_path)
        self.loaded_datasets = loaded_datasets
        self.coefficients = coefficients
        self.use_quad = use_quad
        self.npts = self.loaded_datasets.Commondata.size
        self.single_parameter_fits = single_parameter_fits
        self.use_multiplicative_prescription = use_multiplicative_prescription
        self.external_chi2 = external_chi2
        self.counter = 0

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

    def load_external_chi2(self):
        chi2_modules = {}
        for name, chi2_mod_path in self.external_chi2.items():
            path = pathlib.Path(chi2_mod_path)
            base_path, stem = path.parent, path.stem
            sys.path = [str(base_path)] + sys.path
            chi2_modules[name] = importlib.import_module(stem)

        return chi2_modules

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

        chi2_tot = chi2.compute_chi2(
            self.loaded_datasets,
            self.coefficients.value,
            self.use_quad,
            self.use_multiplicative_prescription,
            use_replica,
        )

        # add external chi2 modules if specified in the runcard
        if self.external_chi2 is not None:
            chi2_modules = self.load_external_chi2()
            for name, chi2_module in chi2_modules.items():
                chi2_ext = getattr(chi2_modules[name], name)
                chi2_tot += chi2_ext(self.coefficients)

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
