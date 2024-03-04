# -*- coding: utf-8 -*-
import itertools
import pathlib
from collections.abc import Iterable

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
import matplotlib.patches as patches
from .spider import radar_factory


from .contours_2d import confidence_ellipse, plot_contours, split_solution
from .latex_tools import latex_packages, multicolum_table_header


def get_confidence_values(dist, has_posterior=True):
    """
    Get confidence level bounds given the distribution
    """
    cl_vals = {}
    if has_posterior:
        cl_vals["low68"] = np.nanpercentile(dist, 16)
        cl_vals["high68"] = np.nanpercentile(dist, 84)
        cl_vals["low95"] = np.nanpercentile(dist, 2.5)
        cl_vals["high95"] = np.nanpercentile(dist, 97.5)
        cl_vals["mid"] = np.mean(dist, axis=0)

    else:
        cl_vals["low68"] = dist["68CL"][0]
        cl_vals["high68"] = dist["68CL"][1]
        cl_vals["low95"] = dist["95CL"][0]
        cl_vals["high95"] = dist["95CL"][1]
        cl_vals["mid"] = dist["bestfit"]

    for cl in [68, 95]:
        cl_vals[f"mean_err{cl}"] = (cl_vals[f"high{cl}"] - cl_vals[f"low{cl}"]) / 2.0
        cl_vals[f"err{cl}_low"] = cl_vals["mid"] - cl_vals[f"low{cl}"]
        cl_vals[f"err{cl}_high"] = cl_vals[f"high{cl}"] - cl_vals["mid"]

    return cl_vals


def compute_confidence_level(
    posterior, coeff_info, has_posterior, disjointed_list=None
):
    """
    Compute central value, 95 % and 68 % confidence levels and store the result in a dictionary
    given a posterior distribution
    Parameters
    ----------
        posterior : dict
            posterior distributions per coefficient
        coeff_info : pandas.Series
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
    bounds = {}
    for (group, name), latex_name in coeff_info.items():
        if name not in posterior:
            bounds[(group, latex_name)] = {}
        else:
            posterior[name] = np.array(posterior[name])
            # double soultion
            if name in disjointed_list:
                solution1, solution2 = split_solution(posterior[name])
                bounds[(group, latex_name)] = pd.DataFrame(
                    [get_confidence_values(solution1), get_confidence_values(solution2)]
                ).stack()
            # single solution
            else:
                bounds[(group, latex_name)] = pd.DataFrame(
                    [get_confidence_values(posterior[name], has_posterior)]
                ).stack()

    return pd.DataFrame(bounds)


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
    report_path: pathlib.Path, str
        path to base folder, where the reports will be stored.
    coeff_config : pandas.DataFrame
        coefficients latex names by gropup type
    logo : bool
        if True dispaly the logo on scatter and bar plots

    """

    def __init__(self, report_path, coeff_config, logo=False):
        self.report_folder = report_path
        self.coeff_info = coeff_config

        # SMEFiT logo
        if logo:
            self.logo = plt.imread(
                f"{pathlib.Path(__file__).absolute().parent}/logo.png"
            )
        else:
            self.logo = None

        self.npar = self.coeff_info.shape[0]

    def _plot_logo(self, ax):
        if self.logo is not None:
            ax.imshow(
                self.logo,
                aspect="auto",
                transform=ax.transAxes,
                extent=[0.8, 0.999, 0.001, 0.30],
                zorder=-1,
            )

    def _get_suplblots(self, figsize):
        groups = self.coeff_info.groupby(level=0).count()
        _, axs = plt.subplots(
            groups.size,
            1,
            gridspec_kw={"height_ratios": groups.values},
            figsize=figsize,
        )
        if not isinstance(axs, Iterable):
            axs = np.array([axs])
        return groups, axs

    def plot_coeffs(
        self, bounds, figsize=(10, 15), x_min=-400, x_max=400, x_log=True, lin_thr=1e-1
    ):
        """
        Plot central value + 95% CL errors

        Parameters
        ----------
            bounds: dict
                confidence level bounds per fit and coefficient
                Note: double solutions are appended under "2"
        """
        groups, axs = self._get_suplblots(figsize)
        bas10 = np.concatenate([-np.logspace(-4, 2, 7), np.logspace(-4, 2, 7)])

        # Spacing between fit results
        nfits = len(bounds)
        y_shift = np.linspace(-0.2 * nfits, 0.2 * nfits, nfits)
        colors = cm.get_cmap("tab20")

        def plot_error_bars(ax, vals, cnt, i, label=None):
            ax.errorbar(
                x=vals.mid,
                y=Y[cnt] + y_shift[i],
                xerr=[[vals.err95_low], [vals.err95_high]],
                color=colors(2 * i + 1),
            )
            ax.errorbar(
                x=vals.mid,
                y=Y[cnt] + y_shift[i],
                xerr=[[vals.err68_low], [vals.err68_high]],
                color=colors(2 * i),
                fmt=".",
                label=label,
            )

        # loop on gropus
        cnt_plot = 0
        for ax, (g, npar) in zip(axs, groups.items()):
            Y = 3 * np.arange(npar)
            # loop over fits
            for i, (fit_name, bound_df) in enumerate(bounds.items()):
                label = fit_name
                # loop on coeffs
                for cnt, (coeff_name, coeff) in enumerate(bound_df[g].items()):
                    # maybe there are no double solutions
                    key_not_found = f"{coeff_name} posterior is not found in {fit_name}"
                    try:
                        vals = coeff.dropna()[0]
                    except KeyError as key_not_found:
                        # fitted ?
                        if coeff.dropna().empty:
                            continue
                        raise KeyError from key_not_found
                    plot_error_bars(ax, vals, cnt, i, label=label)
                    label = None
                    # double solution
                    try:
                        vals_2 = coeff.dropna()[1]
                        plot_error_bars(ax, vals_2, cnt, i)
                    except KeyError:
                        pass

            # y thicks
            ax.set_ylim(-2, Y[-1] + 2)
            ax.set_yticks(Y, self.coeff_info[g], fontsize=13)
            # x grid
            ax.vlines(0, -2, Y[-1] + 2, ls="dashed", color="black", alpha=0.7)
            if x_log:
                x_thicks = np.concatenate([bas * np.arange(1, 10) for bas in bas10])
                if isinstance(lin_thr, dict):
                    thr = lin_thr[g]
                else:
                    thr = lin_thr
                x_thicks = x_thicks[np.abs(x_thicks) > thr / 10]
                ax.set_xscale("symlog", linthresh=thr)
                ax.set_xticks(x_thicks, minor=True)
            ax.grid(True, which="both", ls="dashed", axis="x", lw=0.5)
            if isinstance(x_max, dict):
                ax.set_xlim(x_min[g], x_max[g])
            else:
                ax.set_xlim(x_min, x_max)

            ax.set_title(f"\\rm {g}", x=0.95, y=1.0)
            cnt_plot += npar

        self._plot_logo(axs[-1])
        axs[-1].set_xlabel(r"$c_i/\Lambda^2\ ({\rm TeV}^{-2})$", fontsize=20)
        axs[0].legend(loc=0, frameon=False, prop={"size": 13})
        plt.tight_layout()
        plt.savefig(f"{self.report_folder}/coefficient_central.pdf", dpi=500)
        plt.savefig(f"{self.report_folder}/coefficient_central.png")

    def plot_coeffs_bar(
        self,
        error,
        figsize=(10, 15),
        plot_cutoff=400,
        x_log=True,
        x_min=1e-2,
        x_max=500,
        legend_loc="best",
    ):
        """
        Plot error bars at given confidence level

        Parameters
        ----------
            error: dict
               confidence level bounds per fit and coefficient
        """
        df = pd.DataFrame(error)
        groups, axs = self._get_suplblots(figsize)

        for ax, (g, bars) in zip(axs, df.groupby(level=0)):
            bars.droplevel(0).plot(
                kind="barh",
                width=0.6,
                ax=ax,
                legend=None,
                logx=x_log,
                xlim=(x_min, x_max),
                fontsize=13,
            )
            ax.set_title(f"\\rm {g}", x=0.95, y=1.0)
            ax.grid(True, which="both", ls="dashed", axis="x", lw=0.5)

            # Hard cutoff
            if plot_cutoff is not None:
                ax.vlines(
                    plot_cutoff,
                    -2,
                    3 * groups[g] + 2,
                    ls="dashed",
                    color="black",
                    alpha=0.7,
                )

        self._plot_logo(axs[-1])
        axs[-1].set_xlabel(
            r"$95\%\ {\rm Confidence\ Level\ Bounds}\ (1/{\rm TeV}^2)$", fontsize=20
        )
        axs[0].legend(loc=legend_loc, frameon=False, prop={"size": 13})
        plt.tight_layout()
        plt.savefig(f"{self.report_folder}/coefficient_bar.pdf", dpi=500)
        plt.savefig(f"{self.report_folder}/coefficient_bar.png")

    

    def plot_spider(
            self,
            error,
            labels,
            title,
            log_scale=True,
            fontsize=12,
            figsize=(10, 15),
            plot_cutoff=400,
            x_log=True,
            x_min=1e-2,
            x_max=500,
            legend_loc="best",
    ):
        """
        Plot error bars at given confidence level

        Parameters
        ----------
            error: dict
               confidence level bounds per fit and coefficient
        """

        def log_transform(x, delta):
            return np.log10(x) + delta

        df = pd.DataFrame(error)
        
        # check if more than one fit is loaded
        if df.shape[1] < 2:
            print("At least two fits are required for the spider plot atm")

        theta = radar_factory(len(df), frame='circle')

        # normalise to first fit
        delta = 0.3
        ratio = df.iloc[:, 1:].values / df.iloc[:, 0].values.reshape(-1,1) * 100
        if log_scale:
            data = log_transform(ratio, delta)
        else:
            data = ratio
        

        spoke_labels = [op[1] for op in df.index]

        fig = plt.figure(figsize=(9,9))
        rect = [0.05, 0.05, 0.9, 0.9]
        n_axis = 3
        axes = [fig.add_axes(rect, projection='radar') for i in range(n_axis)]

        radial_perc_lines = [1, 5, 10, 20, 40, 60, 80]
        if log_scale:
            radial_perc_lines = log_transform(radial_perc_lines, delta)

        # take first axis as main, the rest only serve to show the remaining percentage axes
        ax = axes[0]
        angles = np.arange(270, 270 + 360, 360./ n_axis) # zero degrees is 12 o'clock
        perc_labels = [r"$1\%$", r"$5\%$", r"$10\%$", r"$20\%$", r"$40\%$", r"$60\%$", r"$80\%$"]

        for i, axis in enumerate(axes):
            if i > 0:
                axis.patch.set_visible(False)
                axis.grid("off")
                axis.xaxis.set_visible(False)

            if i == 0:
                axis.set_rgrids(radial_perc_lines, angle=angles[i], labels=perc_labels)
            else:
                axis.set_rgrids(radial_perc_lines[1:], angle=angles[i], labels=perc_labels[1:])
            axis.yaxis.set_tick_params(labelsize=10)
            if log_scale:
                axis.set_ylim(0, log_transform(110, delta))
            else:
                axis.set_ylim(0, 110)

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        marker = itertools.cycle(('*', '.'))

        for i, data_fit_i in enumerate(data.T):
            ax.plot(theta, data_fit_i, label=labels[i + 1], color=colors[i])
            ax.scatter(theta, data_fit_i, marker= next(marker), s=140, color=colors[i])
            ax.fill(theta, data_fit_i, alpha=0.25, label='_nolegend_', color=colors[i])

        ax.set_varlabels(spoke_labels, fontsize=fontsize)
        ax.set_title(title)


        ax2 = fig.add_axes(rect=[0, 0, 1, 1])
        ax2.patch.set_visible(False)
        ax2.grid("off")
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        radius = 0.5


        class_names = ['4H', '2L2H', '2FB', 'B']
        angle_sweep = [sum([op_type in index for index in self.coeff_info.index]) / len(self.coeff_info) for op_type in class_names]

        # determine angles of the colored arcs
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        filled_start_angle = 90 - 180 / len(self.coeff_info)

        for i, op_type in enumerate(class_names):
              # Start angle in degrees
            filled_end_angle = angle_sweep[i] * 360 + filled_start_angle  # End angle in degrees
            center = (0.5, 0.5)  # Coordinates relative to the figure

            alpha = 0.3

            ax2.axis('off')

            # Create the filled portion of the circular patch
            filled_wedge = patches.Wedge(center, radius, filled_start_angle, filled_end_angle, facecolor=colors[i],
                                         alpha=alpha, ec=None, width=0.05, transform = ax2.transAxes)
            ax2.add_patch(filled_wedge)

            filled_start_angle += angle_sweep[i] * 360

        ax.legend(loc=(0, 1), labelspacing=0, fontsize='small', frameon=False)

        plt.savefig(f"{self.report_folder}/spider_plot.pdf", dpi=500, bbox_inches='tight')
        plt.savefig(f"{self.report_folder}/spider_plot.png", bbox_inches='tight')


    def plot_posteriors(self, posteriors, labels, disjointed_lists=None):
        """Plot posteriors histograms.

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
        fig = plt.figure(figsize=(grid_size * 4, grid_size * 4))
        # loop on coefficients
        for idx, ((_, l), latex_name) in enumerate(self.coeff_info.items()):
            try:
                ax = plt.subplot(grid_size , grid_size, idx + 1)
            except ValueError:

                ax = plt.subplot(grid_size, grid_size, idx + 1)
            # loop on fits
            for clr_idx, posterior in enumerate(posteriors):
                if l not in posterior:
                    continue
                solution = posterior[l]

                if disjointed_lists[clr_idx] is None:
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

        fig.legend(
            lines, labels, ncol=len(posteriors),
            prop={"size": 25 * (grid_size * 4) / 20}, bbox_to_anchor=(0.5, 1.0), loc='upper center', frameon=False
        )
        fig.tight_layout(rect=[0, 0.05 * (5. / grid_size), 1, 1 - 0.08 * (5. / grid_size)])
        #fig.tight_layout()

        plt.savefig(f"{self.report_folder}/coefficient_histo.pdf")
        plt.savefig(f"{self.report_folder}/coefficient_histo.png")

    def plot_contours_2d(
        self,
        posteriors,
        labels,
        confidence_level=95,
        dofs_show=None,
        double_solution=None,
    ):
        """Plots 2D marginalised projections confidence level contours

        Parameters
        ----------
        posteriors : list
            posterior distributions per fit and coefficient
        labels : list
            list of fit names
        dofs_show: list, optional
            List of coefficients to include in the cornerplot, set to ``None`` by default, i.e. all fitted coefficients
            are included.
        """

        if double_solution is None:
            double_solution = {"fit1": [], "fit2": []}
        if dofs_show is not None:
            posteriors = [
                (posterior[0][dofs_show], posterior[1]) for posterior in posteriors
            ]
            coeff = dofs_show
            n_par = len(dofs_show)
        else:
            coeff = self.coeff_info.index.levels[1]
            n_par = self.npar

        n_cols = n_par - 1 if n_par != 2 else 2
        n_rows = n_cols
        if n_par > 2:
            fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))
            grid = plt.GridSpec(n_rows, n_cols, hspace=0.1, wspace=0.1)
        else:
            fig = plt.figure()
            grid = plt.GridSpec(1, 2, hspace=0.1, wspace=0.1)

        c1_old = coeff[0]

        row_idx = -1
        col_idx = -1
        j = 1

        # loop over coefficient pairs
        for c1, c2 in itertools.combinations(coeff, 2):
            if c1 != c1_old:
                row_idx += -1
                col_idx = -1 - j
                j += 1
                c1_old = c1
            if n_par > 2:
                ax = fig.add_subplot(grid[row_idx, col_idx])
            else:
                ax = fig.add_subplot(grid[0, 0])

            # loop over fits
            hndls_all = []
            fit_number = -1
            for clr_idx, (posterior, kde) in enumerate(posteriors):
                fit_number = fit_number + 1
                if isinstance(confidence_level, list):
                    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                    # plot the first one dashed
                    if kde:
                        sns.kdeplot(
                            x=posterior[c2].values,
                            y=posterior[c1].values,
                            levels=[1 - confidence_level[0] / 100.0, 1.0],
                            bw_adjust=1.2,
                            ax=ax,
                            linestyles="dashed",
                            linewidths=2,
                            color=colors[clr_idx],
                            label=confidence_level[0],
                        )
                    else:
                        confidence_ellipse(
                            posterior[c2].values,
                            posterior[c1].values,
                            ax,
                            edgecolor=colors[clr_idx],
                            confidence_level=confidence_level[0],
                            linestyle="dashed",
                            linewidth=2,
                        )
                    cl = confidence_level[1]
                else:
                    cl = confidence_level

                hndls_contours = plot_contours(
                    ax,
                    posterior,
                    coeff1=c1,
                    coeff2=c2,
                    ax_labels=[
                        self.coeff_info[:, c1].values[0],
                        self.coeff_info[:, c2].values[0],
                    ],
                    kde=kde,
                    clr_idx=clr_idx,
                    confidence_level=cl,
                    double_solution=list(double_solution.values())[clr_idx],
                )
                hndls_all.append(hndls_contours)

                if row_idx != -1:
                    ax.set(xlabel=None)
                    ax.tick_params(
                        axis="x",  # changes apply to the x-axis
                        which="both",  # both major and minor ticks are affected
                        labelbottom=False,
                    )
                if (n_par > 2 and col_idx != -n_cols) or (n_par == 2 and col_idx != -1):
                    ax.set(ylabel=None)
                    ax.tick_params(
                        axis="y",  # changes apply to the y-axis
                        which="both",  # both major and minor ticks are affected
                        labelleft=False,
                    )
                if isinstance(confidence_level, list):
                    hndls_all.append(
                        mlines.Line2D(
                            [],
                            [],
                            linestyle="--",
                            linewidth=2,
                            alpha=1,
                            color=colors[clr_idx],
                        )
                    )
                    hndls_all.append(
                        mlines.Line2D(
                            [],
                            [],
                            linestyle="-",
                            linewidth=2,
                            alpha=1,
                            color=colors[clr_idx],
                        )
                    )
                    hndls_all.append(
                        mlines.Line2D(
                            [],
                            [],
                            color=colors[clr_idx],
                            marker="o",
                            linestyle="None",
                            markersize=8,
                            label="Best fit",
                        )
                    )
                    labels.insert(
                        1 + 4 * fit_number, str(confidence_level[0]) + "\% C.L."
                    )
                    labels.insert(
                        2 + 4 * fit_number, str(confidence_level[1]) + "\% C.L."
                    )
                    labels.insert(3 + 4 * fit_number, "Best fit")
                    # labels already contain all fit name, so for each fit we need to insert C.L. contours at right pos
            hndls_sm_point = ax.scatter(0, 0, c="k", marker="+", s=50, zorder=10)
            hndls_all.append(hndls_sm_point)

            col_idx -= 1
            ax.locator_params(axis="x", nbins=5)
            ax.locator_params(axis="y", nbins=6)
            ax.minorticks_on()
            ax.grid(linestyle="dotted", linewidth=0.5)

        ax = fig.add_subplot(grid[0, -1])
        ax.axis("off")
        ax.legend(
            labels=labels + [r"$\mathrm{SM}$"],
            handles=hndls_all,
            loc="upper left",
            frameon=False,
            fontsize=24,
            handlelength=1,
            borderpad=0.5,
            handletextpad=1,
            title_fontsize=24,
        )

        fig.suptitle(
            r"$\mathrm{Marginalised}\:95\:\%\:\mathrm{C.L.\:intervals}$", fontsize=18
        )
        grid.tight_layout(fig)
        fig.savefig(f"{self.report_folder}/contours_2d.pdf")
        fig.savefig(f"{self.report_folder}/contours_2d.png")

    def write_cl_table(self, bounds, round_val=3):
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
            L.append(f"\\multirow{{{coeff_group.shape[0]}}}{{*}}{{{group}}}")
            # loop on coefficients
            for latex_name in coeff_group.values:
                temp = f" & {latex_name}"
                temp2 = " &"
                # loop on fits
                for bound_df in bounds.values():
                    try:
                        cl_vals = bound_df[(group, latex_name)].dropna()[0]
                    except KeyError:
                        # not fitted
                        if bound_df[(group, latex_name)].dropna().empty:
                            temp += r" & \textemdash & \textemdash & \textemdash "
                            continue
                        raise KeyError(f"{latex_name} is not found in posterior")

                    temp += f" & {np.round(cl_vals['mid'],round_val)} \
                            & [{np.round(cl_vals['low68'],round_val)},{np.round(cl_vals['high68'],round_val)}] \
                                & [{np.round(cl_vals['low95'],round_val)},{np.round(cl_vals['high95'],round_val)}]"
                    # double solution
                    try:
                        cl_vals_2 = bound_df[(group, latex_name)].dropna()[1]
                        temp2 += f" & {np.round(cl_vals_2['mid'],round_val)} \
                                & [{np.round(cl_vals_2['low68'],round_val)},{np.round(cl_vals_2['high68'],round_val)}] \
                                    & [{np.round(cl_vals_2['low95'],round_val)},{np.round(cl_vals_2['high95'],round_val)}]"
                    except KeyError:
                        temp2 += r" & & &"

                # append double solution
                if temp2 != " &" * (3 * nfits + 1):
                    temp += f" \\\\ \\cline{{3-{(2 + 3 * nfits)}}}"
                    temp += temp2

                temp += f" \\\\ \\cline{{2-{(2 + 3 * nfits)}}}"
                L.append(temp)
            L.append(r"\hline")
        L.extend(
            [
                r"\end{tabular}",
                r"\caption{Coefficient comparison}",
                r"\end{table}",
                "\n",
                "\n",
            ]
        )
        return L
