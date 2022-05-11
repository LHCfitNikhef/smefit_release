# -*- coding: utf-8 -*-
import numpy as np

from .latex_tools import latex_packages, multicolum_table_header


def add_bounded_coeff(coeff_dict, coeff_df):
    """zip fixed values and coefficients"""
    line = r" & & = "
    for op, val in zip(coeff_dict["fixed"], coeff_dict["value"]):
        latex_name = coeff_df[:, op].values[0]
        sign = "+" if val > 0 else "$-$"
        line += rf"{sign}\ {np.abs(val):3f}\ {latex_name}"
        if isinstance(val, dict):
            for non_lin_op, power in val.items():
                non_lin_name = coeff_df[:, non_lin_op].values[0]
                line += rf"\ {non_lin_name}^{power} "
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

    def __init__(self, fits, data_groups, coeff_config):

        self.fits = fits
        self.data_info = data_groups
        self.coeff_info = coeff_config
        self.nfits = len(self.fits)

    def write(self):
        """
        Collect the summary tables

        Returns
        -------
            L : list(str)
                list of the latex commands
        """
        L = latex_packages()
        L.extend([r"\usepackage{underscore}", r"\begin{document}"])

        L.extend(self.write_fit_settings())
        L.extend(self.write_dataset_table())
        L.extend(self.write_coefficients_table())
        return L

    def write_fit_settings(self):
        """
        Write fit settings

        Returns
        -------
            L : list(str)
                list of the latex commands
        """
        L = []
        for i, fit in enumerate(self.fits):
            if fit.config["use_quad"]:
                eft_order = r"$\Lambda^{-4}$"
            else:
                eft_order = r"$\Lambda^{-2}$"
            L.extend(
                [
                    r"{\bf \underline{Fit%d:}} %s" % (i + 1, fit.label) + "\n",
                    f"Number of replicas/samples: {fit.Nrep}\n",
                    f"pQCD order: {fit.config['order']}\n",
                    f"SMEFT order: {eft_order}\n",
                ]
            )
        return L

    def write_dataset_table(self):
        """Write the summary tables

        Returns
        -------
            L : list(str)
                list of the latex commands
        """
        L = [
            r"\begin{table}[H]",
            r"\footnotesize",
            r"\centering",
            r"\begin{tabular}{|c|l|" + "c|" * self.nfits + "}",
            r"\hline",
        ]
        temp = " Type & Datasets "
        for fit in self.fits:
            temp += f" & {fit.label}"
        temp += r" \\ \hline"
        L.append(temp)

        for group, datasets in self.data_info.groupby(level=0):
            datasets = datasets.droplevel(0)
            L.append(r"\multirow{%d}{*}{%s}" % (datasets.shape[0], group))
            for isub, (dataset, link) in enumerate(datasets.items()):
                temp = r" & \href{" + link + "}{" + dataset + "} "
                for fit in self.fits:
                    temp += " & "
                    if dataset in fit.config["datasets"]:
                        temp += r"\checkmark"
                if isub != datasets.shape[0] - 1:
                    temp += r"\\ \cline{2-%d}" % (2 + self.nfits)
                L.append(temp)
            L.append(r"\\ \hline")

        # TODO: add total number of datapoints
        # temp = r"\hline & Total number of data points"
        # for fit in self.fits:
        #     temp += f" & {np.sum(fit.datasets.NdataExp)}"
        # temp += r" \\ \hline"
        # L.extend(
        #     [temp, r"\end{tabular}", r"\caption{Dataset comparison.}", r"\end{table}"]
        # )
        L.extend([r"\end{tabular}", r"\caption{Dataset comparison.}", r"\end{table}"])
        return L

    def write_coefficients_table(self):
        """Write the coefficients tables

        Returns
        -------
            L : list(str)
                list of the latex commands
        """
        L = [
            r"\begin{table}[H]",
            r"\centering",
            r"\begin{tabular}{|c|c|" + "c|c|" * self.nfits + "}",
        ]
        L.extend(multicolum_table_header([fit.label for fit in self.fits]))
        L.append(
            r"Class & Coefficients" + r" & Fitted & Fixed " * self.nfits + r" \\ \hline"
        )

        # Build lists for grouped coefficients
        n_fitted = np.zeros(self.nfits, dtype=int)

        # loop on coeff groups
        for group, coefficients in self.coeff_info.groupby(level=0):
            coefficients = coefficients.droplevel(0)
            L.append(r"\multirow{%d}{*}{%s}" % (coefficients.shape[0], group))

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
                            temp += r" & & = %d " % (coeff_dict["value"])
                        else:
                            temp += r" & \checkmark & "
                            n_fitted[i] += 1
                    else:
                        temp += r" & & "

                if isub != coefficients.shape[0] - 1:
                    temp += r"\\ \cline{2-%d}" % (2 + 2 * self.nfits)
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
                r"\caption{Coefficient comparison.}",
                r"\end{table}",
            ]
        )
        return L
