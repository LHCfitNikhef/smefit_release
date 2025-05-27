# -*- coding: utf-8 -*-
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rich.progress import track

from .latex_tools import latex_packages
from .pca import impose_constrain


class HandlerTriangle(HandlerPatch):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        center = (width / 2 - xdescent, height / 2 - ydescent)
        size = min(width, height) / 2
        # Define the lower-left triangle vertices

        if orig_handle.xy[0, 0] < orig_handle.xy[1, 0]:
            vertices = [
                (center[0] - size, center[1] - size),  # Bottom-left
                (center[0] + size, center[1] - size),  # Bottom-right
                (center[0] - size, center[1] + size),  # Top-left
            ]
        else:
            # (upper-right)
            vertices = [
                (center[0] + size, center[1] + size),  # Top-right
                (center[0] - size, center[1] + size),  # Top-left
                (center[0] + size, center[1] - size),  # Bottom-right
            ]
        p = mpatches.Polygon(
            vertices,
            closed=True,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
        )
        p.set_transform(trans)
        return [p]


class FisherCalculator:
    """Computes and writes the Fisher information table, and plots heat map.

    Linear Fisher information depends only on the theoretical corrections,
    while quadratic information requires fit results.
    Parameter constraints are also taken into account. Only fitted
    degrees of freedom are shown in the tables.

    Parameters
    ----------
    coefficients: smefit.coefficients.CoefficienManager
        coefficient manager
    datasets: smefit.loader.DataTuple
        DataTuple object with all the data information
    best_fit_point: pandas.DataFrame
        best fit point of the coefficients
    """

    def __init__(self, coefficients, datasets, best_fit_point, compute_quad):
        self.coefficients = coefficients
        self.free_parameters = self.coefficients.free_parameters.index
        self.datasets = datasets
        self.best_fit_point = best_fit_point

        # update eft corrections with the constraints
        if compute_quad:
            (
                self.new_LinearCorrections,
                self.new_QuadraticCorrections,
            ) = impose_constrain(self.datasets, self.coefficients, update_quad=True)
        else:
            self.new_LinearCorrections = impose_constrain(
                self.datasets, self.coefficients
            )

        self.lin_fisher = None
        self.quad_fisher = None
        self.summary_table = None
        self.summary_HOtable = None

    def compute_linear(self):
        """Compute linear Fisher information."""
        fisher_tab = []
        cnt = 0
        for ndat in self.datasets.NdataExp:
            idxs = slice(cnt, cnt + ndat)
            sigma = self.new_LinearCorrections[:, idxs]
            fisher_row = np.diag(sigma @ self.datasets.InvCovMat[idxs, idxs] @ sigma.T)
            fisher_tab.append(fisher_row)
            cnt += ndat
        self.lin_fisher = pd.DataFrame(
            fisher_tab, index=self.datasets.ExpNames, columns=self.free_parameters
        )

    def compute_quadratic(self):
        """Compute quadratic Fisher information."""

        best_fit_point = self.best_fit_point[self.free_parameters].values.flatten()

        quad_symmetrised = 0.5 * (
            np.einsum("ij...->ij...", self.new_QuadraticCorrections)
            + np.einsum("ij...->ji...", self.new_QuadraticCorrections)
        )
        covmat = self.datasets.CovMat

        A = self.new_LinearCorrections + 2 * np.einsum(
            "l, ilm -> im", best_fit_point, quad_symmetrised
        )

        fisher_quad_all = np.einsum("im, mn, jn", A, covmat, A)
        quad_fisher = []
        cnt = 0

        # this neglects correlations across datasets
        for ndat in self.datasets.NdataExp:
            idxs = slice(cnt, cnt + ndat)
            invcovmat_dataset = np.linalg.inv(covmat[idxs, idxs])
            fisher_dataset = np.einsum(
                "im, mn, jn", A[:, idxs], invcovmat_dataset, A[:, idxs]
            )
            quad_fisher.append(np.diag(fisher_dataset))
            cnt += ndat

        self.quad_fisher = pd.DataFrame(
            quad_fisher,
            index=self.datasets.ExpNames,
            columns=self.free_parameters,
        )

    @staticmethod
    def normalize(table, norm, log):
        """
        Normalize a Pandas DataFrame

        Parameters
        ----------
        table: pandas.DataFrame
            table to normalize
        norm: "data", "coeff"
            if "data" it normalize by columns, if "coeff" by rows
        log: bool
            presents the log of the Fisher if True

        Returns
        -------
        pandas.DataFrame
                normalized table

        """
        if table is None or table.empty:
            return None
        if norm == "data":
            axis_sum, axis_div = 1, 0
        elif norm == "coeff":
            axis_sum, axis_div = 0, 1
        else:
            raise ValueError(f"Invalid norm value: {norm}. Must be 'data' or 'coeff'.")

        table = table.div(table.sum(axis=axis_sum), axis=axis_div) * 100
        if log:
            table = np.log(table[table > 0.0])
        return table.replace(np.nan, 0.0)

    def groupby_data(self, table, data_groups, norm, log):
        """Merge fisher per data group."""
        summary_table = pd.merge(
            data_groups.reset_index(), table, left_on="level_1", right_index=True
        )
        summary_table = summary_table.groupby("level_0").sum(numeric_only=True)
        summary_table.index.name = "data_group"

        return self.normalize(summary_table, norm, log)

    def write_grouped(self, coeff_config, data_groups, summary_only):
        """Write Fisher information tables in latex, both for grouped data and for summary.

        Parameters
        ----------
        coeff_config: dict
            coefficient dictionary per group with latex names
        data_groups: dict
            dictionary with datasets  per group and relative links
        summary_only: bool
            if True only the summary Fisher table fro grouped data is written

        Returns
        -------
        list(str)
            list of the latex commands

        """
        L = latex_packages()
        L.extend(
            [
                r"\begin{document}",
                r"\begin{landscape}",
            ]
        )

        # fisher tables per data_group
        if not summary_only:
            for data_group, data_dict in data_groups.groupby(level=0):
                temp_table = self.lin_fisher.loc[data_dict.index.get_level_values(1)]
                temp_HOtable = None

                if self.quad_fisher is not None:
                    temp_HOtable = self.quad_fisher.loc[
                        data_dict.index.get_level_values(1)
                    ]
                L.extend(
                    self._write(
                        temp_table,
                        temp_HOtable,
                        coeff_config,
                        data_dict.droplevel(0),
                        data_group,
                    )
                )
        L.extend(
            self._write(
                self.summary_table,
                self.summary_HOtable,
                coeff_config,
            )
        )
        L.append(r"\end{landscape}")
        return L

    def _write(
        self, lin_fisher, quad_fisher, coeff_config, data_dict=None, data_group=None
    ):
        """Write Fisher information table in latex.

        Parameters
        ----------
        lin_fisher: pandas.DataFrame
            linear Fisher information table
        quad_fisher: pandas.DataFrame, None
            quadratic Fisher information table, None if linear only
        coeff_config: dict
            coefficient dictionary per group with latex names
        data_dict: dict, optional
            dictionary with datasets and relative links
        data_group: str, optional
            data group name

        Returns
        -------
        list(str)
            list of the latex commands

        """

        def color(value, thr_val=10):
            if value > thr_val:
                return ("blue", value)
            return ("black", value)

        L = [
            r"\begin{table}[H]",
            r"\scriptsize",
            r"\centering",
            r"\begin{tabular}{|c|c|" + "c|" * lin_fisher.shape[0] + "}",
            r"\hline",
            f"\\multicolumn{{2}}{{|c|}}{{}} \
                & \\multicolumn{{{lin_fisher.shape[0]}}}{{c|}}{{Processes}} \\\\ \\hline",
        ]
        temp = " Class & Coefficient "
        if data_dict is None:
            for dataset in lin_fisher.index:
                temp += f"& {{\\rm {dataset} }}"
        else:
            for dataset, link in data_dict.items():
                temp += f"& \\href{{{link}}}{{${{\rm {dataset}}}$}}".replace("_", r"\_")
        temp += r"\\ \hline"
        L.append(temp)

        # loop on coeffs
        for coeff_group, coeff_dict in coeff_config.groupby(level=0):
            coeff_dict = coeff_dict.droplevel(0)
            L.append(f"\\multirow{{{coeff_dict.shape[0]}}}{{*}}{{{coeff_group}}}")
            for coeff, latex_name in coeff_dict.items():
                idx = np.where(coeff == self.free_parameters)[0][0]
                temp = f" & {latex_name}"

                # loop on columns
                for idj, fisher_col in enumerate(lin_fisher.values):
                    temp += r" & \textcolor{%s}{%.2f}" % color(fisher_col[idx])
                    if quad_fisher is not None:
                        temp += r"(\textcolor{%s}{%0.2f})" % color(
                            quad_fisher.iloc[idj, idx]
                        )
                temp += (
                    r"\\ \hline"
                    if coeff == [*coeff_dict.keys()][-1]
                    else f"\\\\ \\cline{{2-{(2 + lin_fisher.shape[0])}}}"
                )
                L.append(temp)

        caption = (
            "Fisher information"
            if data_group is None
            else f"Fisher information in {data_group} datasets"
        )
        L.extend(
            [
                r"\end{tabular}",
                f"\\caption{{{caption}}}",
                r"\end{table}",
            ]
        )
        return L

    @staticmethod
    def unify_fishers(df, df_other):

        if df_other is None or df is None:
            return None

        # Get the union of row and column indices
        all_rows = df.index.union(df_other.index)
        all_columns = df.columns.union(df_other.columns)

        # Reindex both DataFrames to have the same rows and columns
        df = df.reindex(index=all_rows, columns=all_columns, fill_value=0)
        df_other = df_other.reindex(index=all_rows, columns=all_columns, fill_value=0)

        return df, df_other

    @staticmethod
    def set_ticks(ax, yticks, xticks, latex_names, x_labels):
        ax.set_yticks(yticks, labels=latex_names[::-1], fontsize=16)
        ax.set_xticks(
            xticks,
            labels=x_labels,
            rotation=90,
            fontsize=22,
        )
        ax.xaxis.set_ticks_position("top")
        ax.tick_params(which="major", top=False, bottom=False, left=False)
        ax.set_xticks(xticks - 0.5, minor=True)
        ax.set_yticks(yticks - 0.5, minor=True)
        ax.tick_params(which="minor", bottom=False)
        ax.grid(visible=True, which="minor", alpha=0.2)

    @staticmethod
    def plot_values(ax, dfs, cmap, norm, labels=None):
        """
        Plot the values of the Fisher information.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            axes object
        dfs: list
            list of pandas.DataFrame
        cmap: matplotlib.colors.LinearSegmentedColormap
            colour map
        norm: matplotlib.colors.BoundaryNorm
            normalisation of colorbar
        labels: list, optional
            label elements for legend
        """

        df_1 = dfs[0]
        df_2 = dfs[1] if len(dfs) > 1 else None
        cols, rows = df_1.shape

        # Initialize the delta shift for text positioning
        delta_shift = 0

        for i, row in enumerate(df_1.values.T):
            for j, elem_1 in enumerate(row):

                # start filling from the top left corner
                x, y = j, rows - 1 - i
                ec_1 = "black"

                # if two fishers must be plotted together
                if df_2 is not None:

                    elem_2 = df_2.values.T[i, j]

                    # move position numbers
                    delta_shift = 0.2

                    # highlight operators that exist in one but not the other
                    ec_1 = "C1" if elem_2 == 0 and elem_1 > 0 else "black"

                    if elem_2 > 0:
                        ax.text(
                            x + delta_shift,
                            y + delta_shift,
                            f"{elem_2:.1f}",
                            va="center",
                            ha="center",
                            fontsize=9,
                        )

                        # Create a triangle patch for the second element
                        triangle2 = Polygon(
                            [
                                [x + 0.5, y - 0.5],
                                [x + 0.5, y + 0.5],
                                [x - 0.5, y + 0.5],
                            ],
                            closed=True,
                            facecolor=cmap(norm(elem_2)),
                            edgecolor="black",
                        )
                        ax.add_patch(triangle2)

                if elem_1 > 0:

                    ax.text(
                        x - delta_shift,
                        y - delta_shift,
                        f"{elem_1:.1f}",
                        va="center",
                        ha="center",
                        fontsize=9,
                    )
                    if df_2 is not None:

                        # Create a triangle patch for the first element
                        triangle1 = Polygon(
                            [
                                [x - 0.5, y - 0.5],
                                [x + 0.5, y - 0.5],
                                [x - 0.5, y + 0.5],
                            ],
                            closed=True,
                            facecolor=cmap(norm(elem_1)),
                            edgecolor=ec_1,
                        )
                        ax.add_patch(triangle1)

                        # Create legend elements for the patches
                        legend_elements = [
                            mpatches.Polygon(
                                [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]],
                                closed=True,
                                fc="none",
                                edgecolor="black",
                                label=labels[0],
                            ),
                            mpatches.Polygon(
                                [[0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
                                closed=True,
                                fc="none",
                                edgecolor="black",
                                label=labels[1],
                            ),
                        ]
                        # Add the legend to the plot
                        ax.legend(
                            handles=legend_elements,
                            loc="upper center",
                            fontsize=25,
                            frameon=False,
                            ncol=2,
                            handler_map={mpatches.Polygon: HandlerTriangle()},
                            bbox_to_anchor=(0.5, -0.02),
                        )
                    else:
                        # Create a rectangle patch for the first element
                        rectangle = Polygon(
                            [
                                [x - 0.5, y - 0.5],
                                [x + 0.5, y - 0.5],
                                [x + 0.5, y + 0.5],
                                [x - 0.5, y + 0.5],
                            ],
                            closed=True,
                            ec="grey",
                            color=cmap(norm(elem_1)),
                        )
                        ax.add_patch(rectangle)

        # Set the x and y limits of the plot
        ax.set_xlim(0, cols - 0.5)
        ax.set_ylim(0, rows - 0.5)
        # Set the aspect ratio of the plot to be equal
        ax.set_aspect("equal", adjustable="box")

    def plot_heatmap(
        self,
        latex_names,
        fig_name,
        title=None,
        other=None,
        summary_only=True,
        figsize=(11, 15),
        labels=None,
        column_names=None,
    ):

        fisher_df = self.summary_table if summary_only else self.lin_fisher
        quad_fisher_df = self.summary_HOtable if summary_only else self.quad_fisher

        if other is not None:

            title = ""
            fisher_df_other = other.summary_table if summary_only else other.lin_fisher
            quad_fisher_df_other = (
                other.summary_HOtable if summary_only else other.quad_fisher
            )
            # unify the fisher tables and fill missing values by zeros
            fisher_dfs = self.unify_fishers(fisher_df, fisher_df_other)

            # reshuffle the tables according to the latex names ordering
            fisher_dfs = [
                fisher[latex_names.index.get_level_values(level=1)]
                for fisher in fisher_dfs
            ]

            if quad_fisher_df is not None:
                quad_fisher_dfs = self.unify_fishers(
                    quad_fisher_df, quad_fisher_df_other
                )

                # reshuffle the tables according to the latex names ordering
                quad_fisher_dfs = [
                    fisher[latex_names.index.get_level_values(level=1)]
                    for fisher in quad_fisher_dfs
                ]

        else:
            fisher_dfs = [fisher_df[latex_names.index.get_level_values(level=1)]]
            if quad_fisher_df is not None:
                quad_fisher_dfs = [
                    quad_fisher_df[latex_names.index.get_level_values(level=1)]
                ]

        # reshuffle column name ordering
        if column_names is not None:
            custom_ordering = [list(column.keys())[0] for column in column_names]
            fisher_dfs = [fisher_df.loc[custom_ordering] for fisher_df in fisher_dfs]
            if quad_fisher_df is not None:
                quad_fisher_dfs = [
                    quad_fisher_df.loc[custom_ordering]
                    for quad_fisher_df in quad_fisher_dfs
                ]
            x_labels = [list(column.values())[0] for column in column_names]
        else:
            x_labels = [
                f"\\rm{{{name}}}".replace("_", "\\_") for name in fisher_df.index
            ]

        # colour map
        cmap_full = plt.get_cmap("Blues")
        cmap = colors.LinearSegmentedColormap.from_list(
            f"trunc({{{cmap_full.name}}},{{0}},{{0.8}})",
            cmap_full(np.linspace(0, 0.8, 100)),
        )
        norm = colors.BoundaryNorm(np.arange(110, step=10), cmap.N)

        fig = plt.figure(figsize=figsize)
        if quad_fisher_df is not None:
            ax = fig.add_subplot(121)
        else:
            ax = plt.gca()

        self.plot_values(ax, fisher_dfs, cmap, norm, labels)

        self.set_ticks(
            ax,
            np.arange(fisher_dfs[0].shape[1]),
            np.arange(fisher_dfs[0].shape[0]),
            latex_names,
            x_labels,
        )
        ax.set_title(r"\rm Linear", fontsize=22, y=-0.04)
        cax1 = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.5)
        colour_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax1)
        colour_bar.ax.tick_params(labelsize=22)

        if quad_fisher_df is not None:
            ax = fig.add_subplot(122)
            self.plot_values(ax, quad_fisher_dfs, cmap, norm, labels)

            self.set_ticks(
                ax,
                np.arange(quad_fisher_dfs[0].shape[1]),
                np.arange(quad_fisher_dfs[0].shape[0]),
                latex_names,
                x_labels,
            )
            ax.set_title(r"\rm Quadratic", fontsize=22, y=-0.04)
            cax1 = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.5)
            colour_bar = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax1
            )
            colour_bar.ax.tick_params(labelsize=22)

        # fig.subplots_adjust(top=0.9)

        colour_bar.set_label(
            r"${\rm Normalized\ Value}$",
            fontsize=28,
            labelpad=30,
            rotation=270,
        )

        plt.suptitle(f"\\rm Fisher\\ information\\ {title}", fontsize=25, y=0.98)

        plt.savefig(f"{fig_name}.pdf")
        plt.savefig(f"{fig_name}.png")
