# -*- coding: utf-8 -*-
import pathlib

from rich.console import Console
from rich.style import Style
from rich.table import Table

from .. import chi2
from ..loader import get_dataset


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

    print_rate = 10

    # TODO: docstring

    def __init__(self, results_path, loaded_datasets, coefficients, use_quad):
        self.results_path = pathlib.Path(results_path)
        self.loaded_datasets = loaded_datasets
        self.coefficients = coefficients
        self.use_quad = use_quad
        self.npts = self.loaded_datasets.Commondata.size

        self.counter = 0

    @property
    def free_parameters(self):
        """Returns the free parameters entering fit"""
        return self.coefficients.free_parameters

    def generate_chi2_table(self, chi2_dict, chi2_tot):
        r"""Generate log :math:`\Chi^2` table"""
        table = Table(style=Style(color="white"), title_style="bold cyan", title=None)
        table.add_column("Dataset", style="bold green", no_wrap=True)

        table.add_column("Chi^2 /N_dat")
        for name, val in chi2_dict.items():
            table.add_row(str(name), f"{val:.3}")
        table.add_row("Total", f"{(chi2_tot/self.npts):.3}")

        return table

    def chi2_func(self, use_replica=False, print_log=True):
        r"""
        Wrap the math:`\Chi^2` in a function for the optimizer. Pass noise and
        data info as args. Log the math:`\Chi^2` value and values of the coefficients.

        Returns
        -------
            current_chi2 : np.ndarray
                computed :math:`\Chi^2`
        """
        self.counter += 1
        if print_log:
            print_log = (self.counter % self.print_rate) == 0

        chi2_tot = chi2.compute_chi2(
            self.loaded_datasets,
            self.coefficients.value,
            self.use_quad,
            use_replica,
        )

        if print_log:
            chi2_dict = {}
            for data_name in self.loaded_datasets.ExpNames:
                dataset = get_dataset(self.loaded_datasets, data_name)
                chi2_dict[data_name] = (
                    chi2.compute_chi2(
                        dataset,
                        self.coefficients.value,
                        self.use_quad,
                        use_replica,
                    )
                    / dataset.NdataExp
                )
            console = Console()
            console.print(self.generate_chi2_table(chi2_dict, chi2_tot))

        return chi2_tot
