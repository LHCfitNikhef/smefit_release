# -*- coding: utf-8 -*-
import pathlib

from rich.console import Console
from rich.style import Style
from rich.table import Table

from .. import chi2

print_rate = 5000


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
        table = Table(style=Style(color="white"), title_style="bold cyan", title=None)
        table.add_column("Dataset", style="bold green", no_wrap=True)
        table.add_column("Chi^2 /N_dat")
        for name, val in chi2_dict.items():
            table.add_row(str(name), f"{val:.3}")
        table.add_row("Total", f"{(chi2_tot/self.npts):.3}")
        return table

    def chi2_func(self, use_replica=False):
        r"""
        Wrap the math:`\Chi^2` in a function for the optimizer. Pass noise and
        data info as args. Log the math:`\Chi^2` value and values of the coefficients.

        Returns
        -------
            current_chi2 : np.ndarray
                computed :math:`\Chi^2`
        """
        self.counter += 1
        print_log = (self.counter % print_rate) == 0

        if print_log:
            chi2_tot, chi2_dict = chi2.compute_chi2(
                self.loaded_datasets, self.coefficients.value, self.use_quad, use_replica, True
            )
            console = Console()
            console.print(self.generate_chi2_table(chi2_dict, chi2_tot))
        else:
            chi2_tot = chi2.compute_chi2(
                self.loaded_datasets,
                self.coefficients.value,
                self.use_quad,
                use_replica,
            )
        return chi2_tot
