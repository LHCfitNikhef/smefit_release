# -*- coding: utf-8 -*-
import pathlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

from .latex_tools import latex_packages, multicolum_table_header


def get_confidence_values(dist):
    """
    Get confidence level bounds given the distribution
    """
    cl_vals = {}
    cl_vals["low68"] = np.nanpercentile(dist, 16)
    cl_vals["high68"] = np.nanpercentile(dist, 84)
    cl_vals["low95"] = np.nanpercentile(dist, 2.5)
    cl_vals["high95"] = np.nanpercentile(dist, 97.5)
    cl_vals["mid"] = np.mean(dist, axis=0)
    for cl in [68, 95]:
        cl_vals[f"mean_err{cl}"] = (cl_vals[f"high{cl}"] - cl_vals[f"low{cl}"]) / 2.0
        cl_vals[f"err{cl}_low"] = cl_vals["mid"] - cl_vals[f"low{cl}"]
        cl_vals[f"err{cl}_high"] = cl_vals[f"high{cl}"] - cl_vals["mid"]

    return cl_vals


def split_solution(full_solution):
    """Split the posterior solution"""

    min_val = min(full_solution)
    max_val = max(full_solution)
    mid = np.mean([max_val, min_val])

    # solution 1 should be closer to 0
    solution1 = full_solution[full_solution < mid]
    solution2 = full_solution[full_solution > mid]
    if min(abs(solution2)) < min(abs(solution1)):
        solution1, solution2 = solution2, solution1

    return solution1, solution2


def compute_confidence_level(posterior, coeff_df, outfile, disjointed_list=None):
    """
    Compute central value, 95 % and 68 % confidence levels and store the result in a dictionary
    given a posterior distribution
    Parameters
    ----------
        posterior : dict
            posterior distributions per coefficient
        coeff_df : pandas.DataFrame
            coefficients list for which the bounds are computed with latex names
        disjointed_list: list, optional
            list of coefficients with double solutions

    Returns
    -------
        bounds: pandas.DataFrame
            confidence level bounds per coefficient
            Note: double solutions are appended under "2"
    """

    disjointed_list = disjointed_list or []
    names = []
    bounds = {}
    for name, row in coeff_df.iterrows():
        latex_name = row["latex_name"]
        names.append(name)
        if name not in posterior:
            bounds[latex_name] = {}
        else:
            posterior[name] = np.array(posterior[name])
            # double soultion
            if name in disjointed_list:
                solution1, solution2 = split_solution(posterior[name])
                bounds[latex_name] = pd.DataFrame(
                    [get_confidence_values(solution1), get_confidence_values(solution2)]
                ).stack()
            # single solution
            else:
                bounds[latex_name] = pd.DataFrame(
                    [get_confidence_values(posterior[name])]
                ).stack()

    df = pd.DataFrame(bounds)
    res = {}

    for name, latex_name in zip(names, df.columns):
        res[name] = {}
        res[name]["texname"] = latex_name
        info = list(df[latex_name].values)
        res[name]["central value"] = info[4]
        res[name]["68CL"] = [info[0], info[1]]
        res[name]["95CL"] = [info[2], info[3]]

    with open(outfile, "w") as f:
        json.dump(res, f)

    return df


class CoefficientsPlotter:
    """
    Plots central values + 95% CL errors, 95% CL bounds,
    probability distributions, residuals,
    residual distribution, and energy reach.

    Also writes a table displaying values for
    68% CL bounds and central value + 95% errors.

    Takes into account parameter constraints and displays
    all non-zero parameters.

    Note: coefficients that are known to have disjoint
    probability distributions (i.e. multiple solutions)
    are manually separated by including the coefficient name
    in disjointed_list for disjointed_list2
    for global and single fit results, respectively.

    Parameters
    ----------

    """

    def __init__(self, report_path, free_coeff_config, logo=False):

        self.report_folder = report_path
        self.coeff_info = free_coeff_config

        # SMEFiT logo
        if logo:
            self.logo = plt.imread(
                f"{pathlib.Path(__file__).absolute().parent}/logo.png"
            )
        else:
            self.logo = None

        # coeff list contains the coefficients that are fitted
        # in at least one fit included in the report
        # TODO: this should be already checked, can be simplified
        # flat object to search for latex names
        coeff_df = self.coeff_info.reset_index()[["level_1", 0]]
        coeff_df = coeff_df.set_index("level_1")
        coeff_df.index.name = "name"
        coeff_df = coeff_df.rename(columns={0: "latex_name"})
        self.coeff_df = coeff_df

        self.npar = self.coeff_info.shape[0]

    def plot_logo(self, ax):
        if self.logo is not None:
            ax.imshow(
                self.logo,
                aspect="auto",
                transform=ax.transAxes,
                extent=[0.70, 0.975, 0.80, 0.975],
                zorder=-1,
            )

    def plot_coeffs(
        self, bounds, figsize=(15, 6), y_min=-400, y_max=400, y_log=True, lin_thr=1e-1
    ):
        """
        Plot central value + 95% CL errors

        Parameters
        ----------
            bounds: dict
                confidence level bounds per fit and coefficient
                Note: double solutions are appended under "2"
        """

        plt.figure(figsize)
        ax = plt.subplot(111)

        # X-axis
        X = 2 * np.arange(self.npar)
        nfits = len(bounds)
        # Spacing between fit results
        x_shift = np.linspace(-0.1 * nfits, 0.1 * nfits, nfits)
        colors = cm.get_cmap("tab20")

        def plot_error_bars(ax, vals, cnt, i, label=None):
            ax.errorbar(
                X[cnt] + x_shift[i],
                y=vals.mid,
                yerr=[[vals.err95_low], [vals.err95_high]],
                color=colors(2 * i + 1),
            )
            ax.errorbar(
                X[cnt] + x_shift[i],
                y=vals.mid,
                yerr=[[vals.err68_low], [vals.err68_high]],
                color=colors(2 * i),
                fmt=".",
                label=label,
            )

        # loop over fits
        for i, (name, bound_df) in enumerate(bounds.items()):
            label = name
            for cnt, coeff in enumerate(bound_df):
                # maybe there are no double solutions
                key_not_found = f"{coeff} posterior is not found in {name}"
                try:
                    vals = bound_df[coeff].dropna()[0]
                except KeyError as key_not_found:
                    # fitted ?
                    if bound_df[coeff].dropna().empty:
                        continue
                    raise KeyError from key_not_found
                plot_error_bars(ax, vals, cnt, i, label=label)
                label = None
                # double solution
                try:
                    vals_2 = bound_df[coeff].dropna()[1]
                    plot_error_bars(ax, vals_2, cnt, i)
                except KeyError:
                    pass

        self.plot_logo(ax)
        plt.plot(
            np.arange(-1, 2 * X.size + 2), np.zeros(2 * X.size + 3), "k--", alpha=0.7
        )

        if y_log:
            plt.yscale("symlog", linthresh=lin_thr)
        plt.ylim(y_min, y_max)
        plt.ylabel(r"$c_i/\Lambda^2\ ({\rm TeV}^{-2})$", fontsize=25)

        plt.xlim(-1, (self.npar) * 2 - 1)
        plt.tick_params(which="major", direction="in")
        plt.xticks(X, self.coeff_df["latex_name"], fontsize=15, rotation=45)

        plt.legend(loc=0, frameon=False, prop={"size": 13})
        plt.tight_layout()
        plt.savefig(f"{self.report_folder}/Coeffs_Central.pdf", dpi=500)

    def plot_coeffs_bar(
        self,
        error,
        figsize=(12, 6),
        plot_cutoff=400,
        y_log=True,
        y_min=1e-2,
        y_max=500,
        legend_loc="best",
    ):
        """
        Plot error bars at given confidence level

        Parameters
        ----------
            error: dict
               confidence level bounds per fit and coefficient
        """

        plt.figure(figsize)
        df = pd.DataFrame(error)
        ax = df.plot(kind="bar", rot=0, width=0.6)

        # Hard cutoff
        if plot_cutoff is not None:
            ax.plot(
                np.linspace(-1, 2 * self.npar + 1, 2),
                plot_cutoff * np.ones(2),
                "k--",
                alpha=0.7,
                lw=2,
            )

        self.plot_logo(ax)
        plt.xticks(fontsize=10, rotation=45)
        plt.tick_params(axis="y", direction="in", labelsize=15)
        if y_log:
            plt.yscale("log")
        plt.ylabel(
            r"$95\%\ {\rm Confidence\ Level\ Bounds}\ (1/{\rm TeV}^2)$", fontsize=11
        )
        plt.ylim(y_min, y_max)
        plt.legend(loc=legend_loc, frameon=False, prop={"size": 11})
        plt.tight_layout()
        plt.savefig(f"{self.report_folder}/Coeffs_Bar.pdf", dpi=500)

    def plot_posteriors(self, posteriors, labels, disjointed_lists=None):
        """ " Plot posteriors histograms

        Parameters
        ----------
            posteriors : list
                posterior distributions per fit and coefficient
            labels : list
                list of fit names
            disjointed_list: list, optional
                list of coefficients with double solutions per fit
        """
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        grid_size = int(np.sqrt(self.npar)) + 1
        fig = plt.figure(figsize=(grid_size * 4, grid_size * 3))
        # loop on coefficients
        for idx, (l, row) in enumerate(self.coeff_df.iterrows()):
            latex_name = row["latex_name"]
            ax = plt.subplot(grid_size, grid_size, idx + 1)
            # loop on fits
            for clr_idx, posterior in enumerate(posteriors):
                if l not in posterior:
                    continue
                solution = posterior[l]
                if disjointed_lists is None:
                    pass
                elif l in disjointed_lists[clr_idx]:
                    solution, solution2 = split_solution(posterior[l])
                    ax.hist(
                        solution2,
                        bins="fd",
                        density=True,
                        color=colors[clr_idx],
                        edgecolor="black",
                        alpha=0.3,
                    )
                ax.hist(
                    solution,
                    bins="fd",
                    density=True,
                    color=colors[clr_idx],
                    edgecolor="black",
                    alpha=0.3,
                    label=labels[clr_idx],
                )
                ax.text(
                    0.05,
                    0.85,
                    latex_name,
                    transform=ax.transAxes,
                    fontsize=25,
                )

                ax.tick_params(which="both", direction="in", labelsize=22.5)
                ax.tick_params(labelleft=False)

        lines, labels = fig.axes[0].get_legend_handles_labels()
        for axes in fig.axes:
            if len(axes.get_legend_handles_labels()[0]) > len(lines):
                lines, labels = axes.get_legend_handles_labels()
        fig.legend(lines, labels, loc="lower right", prop={"size": 20})
        plt.tight_layout()
        plt.savefig(f"{self.report_folder}/Coeffs_Hist.pdf")

    def write_cl_table(self, bounds):
        """Coefficients latex table"""
        nfits = len(bounds)
        L = latex_packages()
        L.extend(
            [
                r"\begin{document}",
                "\n",
                "\n",
                r"\begin{table}[H]",
                r"\centering",
                r"\begin{tabular}{|c|c|" + "c|c|c|" * nfits + "}",
            ]
        )
        L.extend(multicolum_table_header(bounds.keys(), ncolumn=3))
        L.append(
            r"Class & Coefficients"
            + r" & best & 68\% CL Bounds & 95\% CL Bounds" * nfits
            + r"\\ \hline"
        )

        for group, coeff_group in self.coeff_info.groupby(level=0):
            coeff_group = coeff_group.droplevel(0)
            L.append(r"\multirow{%d}{*}{%s}" % (coeff_group.shape[0], group))
            # loop on coefficients
            for latex_name in coeff_group.values:
                temp = f" & {latex_name}"
                temp2 = " &"
                # loop on fits
                for bound_df in bounds.values():
                    try:
                        cl_vals = bound_df[latex_name].dropna()[0]
                    except KeyError:
                        # not fitted
                        if bound_df[latex_name].dropna().empty:
                            temp += r" & \textemdash & \textemdash & \textemdash "
                            continue
                        raise KeyError(f"{latex_name} is not found in posterior")

                    temp += (
                        r" & {:0.3f} & [{:0.3f},{:0.3f}] & [{:0.3f},{:0.3f}] ".format(
                            cl_vals["mid"],
                            cl_vals["low68"],
                            cl_vals["high68"],
                            cl_vals["low95"],
                            cl_vals["high95"],
                        )
                    )
                    # double solution
                    try:
                        cl_vals_2 = bound_df[latex_name].dropna()[1]
                        temp2 += r" & {:0.3f} & [{:0.3f},{:0.3f}] & [{:0.3f},{:0.3f}] ".format(
                            cl_vals_2["mid"],
                            cl_vals_2["low68"],
                            cl_vals_2["high68"],
                            cl_vals_2["low95"],
                            cl_vals_2["high95"],
                        )
                    except KeyError:
                        temp2 += r" & & &"

                # append double solution
                if temp2 != " &" * (3 * nfits + 1):
                    temp += r" \\ \cline{3-%d}" % (2 + 3 * nfits)
                    temp += temp2

                temp += r" \\ \cline{2-%d}" % (2 + 3 * nfits)
                L.append(temp)
            L.append(r"\hline")
        L.extend(
            [
                r"\end{tabular}",
                r"\caption{Coefficient comparison.}",
                r"\end{table}",
                "\n",
                "\n",
            ]
        )
        return L
