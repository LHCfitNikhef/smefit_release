# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm

from ..coefficients import Coefficient
from .latex_tools import latex_packages, multicolum_table_header


def add_bounded_coeff(coeff_dict, coeff_df):
    """Build constrain formula"""
    line = r" & & = "
    temp_c = Coefficient(
        "temp_c",
        coeff_dict["min"],
        coeff_dict["max"],
        constrain=coeff_dict["constrain"],
        value=coeff_dict.get("value", None),
    )
    if temp_c.constrain is None:
        fact = temp_c.value
        sign = "+" if fact >= 0 else "$-$"
        line += rf"{sign}\ {fact}"
        return line
    for constrain in temp_c.constrain:
        for op, val in constrain.items():
            fact, exp = val
            latex_name = coeff_df[:, op].values[0]
            sign = "+" if fact > 0 else "$-$"
            if exp > 1:
                line += rf"{sign}\ {np.abs(fact):.3f}\ $(${latex_name}$)^{exp}$"
            else:
                line += rf"{sign}\ {np.abs(fact):.3f}\ {latex_name}"
    return line


class SummaryWriter:
    """
    Provides a summary of the fits included in the report: the fit settings,
    fitted parameters (including any parameter constraints), and
    datasets.

    Uses data_references.yaml, data_groups.yaml, and coeff_groups.yaml
    YAML files. The first gives the references used for hyperlinks,
    and the other two organizes the data and parameters into groups
    (top, higgs, etc.).

    Parameters
    ----------

    """

    def __init__(self, fits, data_groups, coeff_config, data_scales):
        self.fits = fits
        self.data_info = data_groups
        self.coeff_info = coeff_config
        self.nfits = len(self.fits)
        # Get names of datasets for each fit
        self.dataset_fits = []
        self.data_scales = data_scales

        for fit in self.fits:
            self.dataset_fits.append([data["name"] for data in fit.config["datasets"]])

    def fit_settings(self):
        """Fit settings table.

        Returns
        -------
        pd.DataFrame:
            table with the most relevant fit settings
        """
        summaries = {}
        for fit in self.fits:
            summary_dict = {}
            summary_dict["EFT order"] = (
                "Qudratic" if fit.config["use_quad"] else "Linear"
            )
            summary_dict["Replicas"] = fit.n_replica
            label = fit.label.replace(r"\ ", "").replace(r"\rm", "")
            summaries[label] = summary_dict
        return pd.DataFrame(summaries)

    def write_dataset_table(self):
        """Write the summary tables

        Returns
        -------
            L : list(str)
                list of the latex commands
        """
        L = latex_packages()
        L.extend(
            [
                r"\usepackage{underscore}",
                r"\begin{document}",
                r"\begin{longtable}{|c|l|" + "c|" * self.nfits + "}",
                r"\hline",
                r"\footnotesize",
            ]
        )
        temp = " Type & Datasets "
        for fit in self.fits:
            temp += f" & {fit.label}"
        temp += r" \\ \hline"
        L.append(temp)

        for group, datasets in self.data_info.groupby(level=0):
            datasets = datasets.droplevel(0)
            L.append(f"\\multirow{{{datasets.shape[0]}}}{{*}}{{{group}}}")
            for isub, (dataset, link) in enumerate(datasets.items()):
                temp = r" & \href{" + link + "}{" + dataset + "} "
                for data in self.dataset_fits:
                    temp += " & "
                    if dataset in data:
                        temp += r"$\checkmark$"
                if isub != datasets.shape[0] - 1:
                    temp += f"\\\\ \\cline{{2-{(2 + self.nfits)}}}"
                L.append(temp)
            L.append(r"\\ \hline")

        # TODO: load dataset to print this info ??
        # temp = r"\hline & Total number of data points"
        # for fit in self.fits:
        #     temp += f" & {np.sum(fit.datasets.NdataExp)}"
        # temp += r" \\ \hline"
        # L.extend(
        #     [temp, r"\end{tabular}", r"\caption{Dataset comparison}", r"\end{table}"]
        # )
        L.extend([r"\caption{Dataset comparison}", r"\end{longtable}"])
        return L

    def write_coefficients_table(self):
        """Write the coefficients tables

        Returns
        -------
            L : list(str)
                list of the latex commands
        """
        L = latex_packages()
        L.extend(
            [
                r"\usepackage{underscore}",
                r"\begin{document}",
                r"\begin{table}[H]",
                r"\centering",
                r"\begin{tabular}{|c|c|" + "c|c|" * self.nfits + "}",
            ]
        )
        L.extend(multicolum_table_header([fit.label for fit in self.fits]))
        L.append(
            r"Class & Coefficients" + r" & Fitted & Fixed " * self.nfits + r" \\ \hline"
        )

        # Build lists for grouped coefficients
        n_fitted = np.zeros(self.nfits, dtype=int)

        # loop on coeff groups
        for group, coefficients in self.coeff_info.groupby(level=0):
            coefficients = coefficients.droplevel(0)
            L.append(f"\\multirow{{{coefficients.shape[0]}}}{{*}}{{{group}}}")

            # loop on coefficients
            for isub, (coeff, latex_name) in enumerate(coefficients.items()):
                temp = f" & {latex_name}"

                # loop on fits
                for i, fit in enumerate(self.fits):
                    if coeff in fit.config["coefficients"]:
                        coeff_dict = fit.config["coefficients"][coeff]
                        if "constrain" in coeff_dict:
                            temp += add_bounded_coeff(coeff_dict, self.coeff_info)
                        elif "value" in coeff_dict:
                            temp += f" & & = {coeff_dict['value']}"
                        else:
                            temp += r" & $\checkmark$ & "
                            n_fitted[i] += 1
                    else:
                        temp += r" & & "

                if isub != coefficients.shape[0] - 1:
                    temp += f"\\\\ \\cline{{2-{(2 + 2 * self.nfits)}}}"
                L.append(temp)

            L.append(r"\\ \hline")

        temp = r"\hline & Number fitted coefficients"

        # closing line
        for n in n_fitted:
            temp += f" & {n} & "
        temp += r" \\ \hline"

        L.extend(
            [
                temp,
                r"\end{tabular}",
                r"\caption{Coefficient comparison}",
                r"\end{table}",
            ]
        )
        return L

    def plot_data_scales(self, path):
        # Collect scales for each dataset in each group
        # Doing it for all the fits
        fits_datagroup_scales = []
        for fit in self.data_scales:
            fit_scales = {}
            for group, datasets in self.data_info.groupby(level=0):
                fit_scales[group] = np.array([])
                datasets = datasets.droplevel(0)
                for dataset, _ in datasets.items():
                    # concatenate the scales for each dataset in the group
                    fit_scales[group] = np.concatenate(
                        (fit_scales[group], fit[dataset])
                    )
            fits_datagroup_scales.append(fit_scales)

        # Now we plot the scales for each fit
        # We plot a heatmap with groups in the x-axis and scales on the y axis
        # The color of each cell will represent the scale count
        # We will have a plot for each fit
        for i, fit_scales in enumerate(fits_datagroup_scales):
            group_names = list(fit_scales.keys())

            raw_min = min(min(scales) for _, scales in fit_scales.items())
            raw_max = max(max(scales) for _, scales in fit_scales.items())
            bins = np.logspace(
                np.log10(raw_min),
                np.log10(raw_max),
                21,
            )

            # Round to 10 if below 300, otherwise round to 100
            bins = np.where(
                bins < 300, np.round(bins / 10) * 10, np.round(bins / 100) * 100
            )

            # Adjust the first and last bin if necessary to ensure coverage
            if bins[0] > raw_min:
                bins[0] = (
                    np.floor(raw_min / 10) * 10
                    if raw_min < 300
                    else np.floor(raw_min / 100) * 100
                )
            if bins[-1] < raw_max:
                bins[-1] = (
                    np.ceil(raw_max / 10) * 10
                    if raw_max < 300
                    else np.ceil(raw_max / 100) * 100
                )

            order = [
                r"$\bar{t}t\bar{t}t + \bar{t}t\bar{b}b$",
                r"$\rm Higgs$",
                r"$\rm LEP$",
                r"$\bar{t}t$",
                r"$\bar{t}tV$",
                r"$t$",
                r"$tV$",
                r"$VV$",
                r"$\mathrm{FCC\textnormal{-}ee\:91\:GeV}$",
                r"$\mathrm{FCC\textnormal{-}ee\:161\:GeV}$",
                r"$\mathrm{FCC\textnormal{-}ee\:240\:GeV}$",
                r"$\mathrm{FCC\textnormal{-}ee\:365\:GeV}$",
            ]

            # Create a dictionary to map order to their indices
            order_index = {name: i for i, name in enumerate(order)}

            # Sort group names by their order index, keeping unmatched names in original order
            sorted_group_names = sorted(
                group_names,
                key=lambda x: order_index.get(
                    x, np.inf
                ),  # Use `np.inf` for unmatched names
            )

            # Prepare the heatmap data
            heatmap_data = []
            for group in sorted_group_names:
                hist, _ = np.histogram(fit_scales[group], bins=bins)
                heatmap_data.append(hist)

            heatmap_data = np.array(heatmap_data)

            # Replace 0 values with empty strings for annotations
            annot_data = np.where(heatmap_data == 0, "", heatmap_data)
            # Define the bins for discrete colorbar (adjust as needed)
            # Manually define the first few boundaries (0, 1, 2)
            boundaries = np.array([0, 1, 2, 5])

            # Append the rest of the boundaries starting from 4 and spaced by 4
            boundaries = np.concatenate(
                [boundaries, np.arange(10, heatmap_data.max() + 10, 10)]
            )
            norm = BoundaryNorm(boundaries, ncolors=256)
            # Plot the heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            heatmap = sns.heatmap(
                heatmap_data,
                annot=annot_data,
                fmt="",
                cmap="Blues",
                ax=ax,
                xticklabels=[f"{int(bins[i + 1])}" for i in range(len(bins) - 1)],
                yticklabels=sorted_group_names,
                cbar_kws={
                    "ticks": boundaries,
                },
                norm=norm,
            )

            cbar = heatmap.collections[0].colorbar
            cbar.set_label("\\# of Data points", fontsize=14)

            # Adjust the x-tick positions to align with bin edges
            xtick_positions = [i for i in range(len(bins))]
            ax.set_xticks(xtick_positions)  # Set tick positions
            ax.set_xticklabels([f"{int(bins[i])}" for i in range(len(bins))])

            ax.set_title(f"Data Scales for {self.fits[i].label}", fontsize=16)
            ax.set_xlabel(
                "Scales [GeV]",
                fontsize=14,
            )
            fig.tight_layout()

            # Save the heatmap
            fig.savefig(f"{path}/scales_{self.fits[i].name}.pdf")
            fig.savefig(f"{path}/scales_{self.fits[i].name}.png")
