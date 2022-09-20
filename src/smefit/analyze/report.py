# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib import rc, use

from ..fit_manager import FitManager
from ..log import logging
from .coefficients_utils import CoefficientsPlotter, compute_confidence_level
from .correlations import plot_correlations
from .latex_tools import combine_plots, latex_packages, run_pdflatex
from .summary import SummaryWriter

_logger = logging.getLogger(__name__)

# global mathplotlib settings
use("PDF")
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", **{"usetex": True, "latex.preamble": r"\usepackage{amssymb}"})


class Report:
    r"""Class to manage the report.
    If :math:`\chi^2`, Fisher or Data vs Theory plots are produced it computes the
    best fit theory predictions.

    Attributes
    ----------
        report: str
            path to report folder
        fits: numpy.ndarray
            array with fits (instances of `smefit.fit_manager.FitManger`) included in the report
        data_info: pandas.DataFrame
            dataset information (references and data groups)
        coeff_info: pandas.DataFrame
            coefficients information (group and latex name)

    Parameters
    ----------
        report_path: pathlib.Path, str
            path to base folder, where the reports will be stored.
        result_path: pathlib.Path, str
            path to base folder, where the results are stored.
        report_config: dict
            dictionary with report configuration, see `/run_cards/analyze/report_runcard.yaml`
            for and example
    """

    def __init__(self, report_path, result_path, report_config):

        self.report = f"{report_path}/{report_config['name']}"
        self.fits = []
        # build the fits labels if needed
        if "fit_labels" not in report_config:
            fit_labels = [
                r"${\rm %s}$" % fit.replace("_", r"\ ")
                for fit in report_config["result_IDs"]
            ]
        else:
            fit_labels = report_config["fit_labels"]
        # Loads fits
        for name, label in zip(report_config["result_IDs"], fit_labels):
            fit = FitManager(result_path, name, label)
            fit.load_results()
            self.fits.append(fit)
        self.fits = np.array(self.fits)

        # Loads useful information about data
        self.data_info = self._load_grouped_info(report_config["data_info"], "datasets")
        # Loads coefficients grouped with latex name
        self.coeff_info = self._load_grouped_info(
            report_config["coeff_info"], "coefficients"
        )

    def _load_grouped_info(self, raw_dict, key):
        """
        Load grouped info of coefficients and datasets
        Only elements appearing ad lest once in the fit configs are
        kept

        Parameters
        ----------
            raw_dict: dict
                raw dictionary with relevant information
            key: "datasets" or "coefficients"
                key to check

        Returns
        _______
            grouped_config: pandas.DataFrame
                table with information by group
        """
        out_dict = {}
        for group, entries in raw_dict.items():
            out_dict[group] = {}
            for val in entries:
                if np.any([val[0] in fit.config[key] for fit in self.fits]):
                    out_dict[group][val[0]] = val[1]

            if len(out_dict[group]) == 0:
                out_dict.pop(group)
        return pd.DataFrame(out_dict).stack().swaplevel()

    def summary(self):
        """
        Summary Table runner.
        """
        lines = SummaryWriter(self.fits, self.data_info, self.coeff_info).write()
        run_pdflatex(self.report, lines, "summary")

    def coefficients(
        self,
        scatter_plot=None,
        confidence_level_bar=None,
        posterior_histograms=True,
        hide_dofs=None,
        show_only=None,
        logo=True,
        table=True,
        double_solution=None,
    ):
        """
        Coefficients plots and table runner.

        Parameters
        ----------
            hide_dofs: list
                list of operator not to display
            show_only: list
                list of all the operator to display, if None all the free dof are presented
            logo: bool
                if True add logo to the plots
            scatter_plot: None, dict
                kwarg confidence level bar plot or None
            confidence_level_bar: None, dict
                kwarg scatter plot or None
            posterior_histograms: bool
                if True plot the posterior distribution for each coefficient
            table: bool, optional
                write the latex confidence level table per coefficient
            double_solution: dict
                operator with double solution per fit
        """

        free_coeff_config = self.coeff_info
        if show_only is not None:
            free_coeff_config = free_coeff_config.loc[:, show_only]
        if hide_dofs is not None:
            free_coeff_config = free_coeff_config.drop(hide_dofs, level=1)

        coeff_plt = CoefficientsPlotter(
            self.report,
            free_coeff_config,
            logo=logo,
        )

        # compute confidence level bounds
        bounds_dict = {}
        for fit in self.fits:
            bounds_dict[fit.label] = compute_confidence_level(
                fit.results,
                coeff_plt.coeff_df,
                double_solution.get(fit.name, None),
            )

        if scatter_plot is not None:
            _logger.info("Plotting : Central values and Confidence Level bounds")
            coeff_plt.plot_coeffs(bounds_dict, **scatter_plot)

        # when we plot the 95% CL we show 95% CL for null solutions.
        # the error coming from a degenerate solution is not taken into account.
        if confidence_level_bar is not None:
            _logger.info("Plotting : Confidence Level error bars")
            bar_cl = confidence_level_bar["confidence_level"]
            confidence_level_bar.pop("confidence_level")
            zero_sol = 0
            coeff_plt.plot_coeffs_bar(
                {
                    name: -bound_df.loc[zero_sol, f"low{bar_cl}"]
                    + bound_df.loc[zero_sol, f"high{bar_cl}"]
                    for name, bound_df in bounds_dict.items()
                },
                **confidence_level_bar,
            )

        if posterior_histograms:
            _logger.info("Plotting : Posterior histograms")
            coeff_plt.plot_posteriors(
                [fit.results for fit in self.fits],
                labels=[fit.label for fit in self.fits],
                disjointed_lists=list((*double_solution.values(),)),
            )
        if table:
            _logger.info("Writing : Confidence level table")
            lines = coeff_plt.write_cl_table(bounds_dict)

        combine_plots(self.report, lines, "coefficient_plots", "coefficients_")

    def correlations(self, hide_dofs=None, thr_show=0.1):
        """Plot coefficients correlation matrix.

        Parameters
        ----------
            hide_dofs: list
                list of operator not to display.
            thr_show: float, None
                minimum threshold value to show.
                If None the full correlation matrix is displayed.
        """

        for fit in self.fits:
            _logger.info(f"Plotting correlations for: {fit.name}")
            coeff_to_keep = fit.coefficients.free_parameters.index
            plot_correlations(
                fit.results[coeff_to_keep],
                latex_names=self.coeff_info.droplevel(0),
                fig_name=f"{self.report}/correlations_{fit.name}.pdf",
                fit_label=fit.label,
                hide_dofs=hide_dofs,
                thr_show=thr_show,
            )

        L = latex_packages()
        L.append(r"\begin{document}")
        combine_plots(
            self.report,
            L,
            "correlation_plots",
            "correlations_",
        )
