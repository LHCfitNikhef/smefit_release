# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .latex_tools import chi2table_header, latex_packages


class Chi2tableCalculator:
    r"""Compute the :math:`\chi^2` for each replica and
    various plots.

    Parameters
    ----------
    data_info: pandas.DataFrame
        datasets information (references and data groups)
    """

    def __init__(self, data_info):

        # dataset info
        self.data_info = data_info

        self.chi2_df_sm = pd.DataFrame()
        self.chi2_df_sm_grouped = pd.DataFrame()

    @staticmethod
    def compute(datasets, smeft_predictions):
        r"""Compute the :math:`\chi^2` for each replica."""
        chi2 = []
        chi2_sm = []
        chi2_rep = []

        diff = datasets.Commondata - np.mean(smeft_predictions, axis=0)
        covmat_diff = datasets.InvCovMat @ diff

        diff_sm = datasets.Commondata - datasets.SMTheory
        covmat_diff_sm = datasets.InvCovMat @ diff_sm

        # do the difference replica by replica and multiply by cov inverse
        diff_rep = (
            np.tile(datasets.Commondata, (smeft_predictions.shape[0], 1))
            - smeft_predictions
        )
        covmat_diff_rep = datasets.InvCovMat @ diff_rep.T

        # Compute per experiment
        cnt = 0
        for ndat_exp in datasets.NdataExp:

            chi2.append(
                np.dot(
                    diff[cnt : cnt + ndat_exp],
                    covmat_diff[cnt : cnt + ndat_exp],
                )
            )
            chi2_sm.append(
                np.dot(
                    diff_sm[cnt : cnt + ndat_exp],
                    covmat_diff_sm[cnt : cnt + ndat_exp],
                )
            )

            # Compute chi2 by replica
            # multiply the second term of the chi2 and
            # take the diagonal (replica left = replica right).
            # here np.einsum is faster than np.diag(a @ b), since
            # a and b are large usually
            chi2_rep.append(
                np.einsum(
                    "ij,ji->i",
                    diff_rep[:, cnt : cnt + ndat_exp],
                    covmat_diff_rep[cnt : cnt + ndat_exp],
                )
            )
            cnt += ndat_exp

        return (
            pd.DataFrame(
                {
                    "ndat": datasets.NdataExp,
                    "chi2": np.array(chi2),
                    "chi2_std": np.std(chi2_rep, axis=1),
                    "chi2_sm": np.array(chi2_sm),
                },
                index=datasets.ExpNames,
            ),
            np.sum(chi2_rep, axis=0) / datasets.Commondata.size,
        )

    def split_table_entries(self, chi2_df):
        r"""Split the :math:`\chi^2` tables entries,
        spelling out the :math:`\chi^2` per each group, dataset
        and normalizing the values.

        Returns
        -------
        dict
            '': :math:`\chi^2` table entries
            'grouped': :math:`\chi^2` table entries per data group
        """
        # reduced chi2
        chi2_df["chi2/ndat"] = chi2_df["chi2"] / chi2_df["ndat"]
        chi2_df["chi2_sm/ndat"] = chi2_df["chi2_sm"] / chi2_df["ndat"]

        # set colors
        chi2_upper = chi2_df["chi2"] + chi2_df["chi2_std"]
        chi2_lower = chi2_df["chi2"] - chi2_df["chi2_std"]

        chi2_df["color"] = "black"
        chi2_df.loc[chi2_df["chi2_sm"] > chi2_upper, "color"] = "blue"
        chi2_df.loc[chi2_df["chi2_sm"] < chi2_lower, "color"] = "red"

        # merge groups with data names
        chi2_df_grouped = pd.merge(
            self.data_info.reset_index(), chi2_df, left_on="level_1", right_index=True
        ).drop([0, "chi2_std"], axis=1)
        chi2_df_grouped = chi2_df_grouped.groupby("level_0").sum()
        chi2_df_grouped.index.name = "data_group"

        # add total values
        chi2_df_grouped = chi2_df_grouped.append(
            pd.Series(chi2_df_grouped.sum(), name="Total")
        )
        chi2_df_grouped["chi2/ndat"] = chi2_df_grouped["chi2"] / chi2_df_grouped["ndat"]
        chi2_df_grouped["chi2_sm/ndat"] = (
            chi2_df_grouped["chi2_sm"] / chi2_df_grouped["ndat"]
        )

        return chi2_df, chi2_df_grouped

    def split_table_entries_sm(self, chi2_dict, chi2_dict_group):
        """Update the chi2_df dict for all included datasets."""
        labels_to_include = ["ndat", "chi2_sm/ndat"]
        for chi2_df, chi2_df_grouped in zip(
            chi2_dict.values(), chi2_dict_group.values()
        ):
            self.chi2_df_sm = self.chi2_df_sm.append(chi2_df[labels_to_include])
            self.chi2_df_sm_grouped = self.chi2_df_sm_grouped.append(
                chi2_df_grouped[labels_to_include]
            )
        self.chi2_df_sm = self.chi2_df_sm[
            ~self.chi2_df_sm.index.duplicated(keep="first")
        ]
        self.chi2_df_sm_grouped = self.chi2_df_sm_grouped.drop_duplicates()

    def write(self, chi2_dict, chi2_dict_group):
        """Write the chi2 latex tables

        Parameters
        ----------
            chi2_dict : dict
                tables computed with compute() method per fit
            chi2_dict_group: dict
                tables computed with compute() method per fit

        Returns
        -------
            L : list(str)
                list of the latex commands
        """
        self.split_table_entries_sm(chi2_dict, chi2_dict_group)
        L = latex_packages()
        L.extend([r"\usepackage{underscore}", r"\begin{document}"])
        L.extend(self.write_chi2(chi2_dict, chi2_dict_group))
        L.extend(["\n", "\n"])
        L.extend(self.write_chi2_summary(chi2_dict_group))
        return L

    def write_chi2(self, chi2_dict, chi2_dict_group):
        """
        Write the chi2 latex tables per each dataset inside each group

        Parameters
        ----------
            chi2_dict : dict
                Chi2 Table entries per each fit
        Returns
        -------
            L : list(str)
                list of the latex commands
        """
        L = [
            r"$\chi^2$ table. Blue color text represents a value that is lower than the SM $\chi^2$ \
            by more than one standard deviation of the $\chi^2$ distribution.\
            Similarly, red color text represents values that are higher than the SM $\chi^2$ by more than one standard deviation.\
            In parenthesis is the total SM $\chi^2$ for the dataset included in the fit."
        ]
        for group, datasets in self.data_info.groupby(level=0):
            L.extend(
                [
                    r"\begin{table}[H]",
                    r"\centering",
                    r"\begin{tabular}{|l|c|c|" + "C{3cm}|" * len(chi2_dict) + "}",
                ]
            )
            L = chi2table_header(L, chi2_dict.keys())

            # loop over datasets
            for dataset, link in datasets.droplevel(0).items():
                temp = (
                    r"\href{{{}}}{{{}}}".format(link, dataset)
                    + f" & {int(self.chi2_df_sm.loc[dataset,'ndat'])}"
                    + f" & {self.chi2_df_sm.loc[dataset,'chi2_sm/ndat'].round(3)}"
                )
                for chi2_df in chi2_dict.values():
                    temp += " & "
                    if dataset in chi2_df.index:
                        temp += r"{{\color{{{}}} {:.3f}}}".format(
                            chi2_df.loc[dataset, "color"],
                            chi2_df.loc[dataset, "chi2/ndat"],
                        )

                temp += r" \\ \hline"
                L.append(temp)

            closing_line = r"\hline Total & & "
            for chi2_df_grouped in chi2_dict_group.values():
                closing_line += " & "
                if group in chi2_df_grouped.index:
                    closing_line += f"{chi2_df_grouped.loc[group, 'chi2/ndat']:.3f} ({chi2_df_grouped.loc[group, 'chi2_sm/ndat']:.3f})"
            closing_line += r" \\ \hline"
            L.extend(
                [
                    closing_line,
                    r"\end{tabular}",
                    r"\caption{$\chi^2$ table for %s data}" % group,
                    r"\end{table}",
                ]
            )
        return L

    def write_chi2_summary(self, chi2_dict_group):
        r"""Summary chi2 table for grouped data.

        Parameters
        ----------
        chi2_dict_group : dict
            :math:`\chi^2` table entries by data group

        Returns
        -------
        list(str)
           list of the latex commands

        """
        L = [
            r"\begin{table}[H]",
            r"\centering",
            r"\begin{tabular}{|l|" + "C{2cm}|c|" * len(chi2_dict_group) + "}",
            r"\hline",
        ]
        temp = r""
        for label in chi2_dict_group:
            temp += r"& \multicolumn{2}{c|}{%s} " % (label)
        temp += r"\\ \hline"
        L.append(temp)
        L.append(
            r"Process "
            + r" & $N_{\rm data}$ & $\chi^2/N_{\rm data}$" * len(chi2_dict_group)
            + r"\\ \hline",
        )

        for group in self.data_info.index.levels[0]:
            temp = f"{group}"
            for chi2_df_grouped in chi2_dict_group.values():
                if group in chi2_df_grouped.index:
                    temp += " & %d & %.3f (%.3f)" % (
                        chi2_df_grouped.loc[group, "ndat"],
                        chi2_df_grouped.loc[group, "chi2/ndat"],
                        chi2_df_grouped.loc[group, "chi2_sm/ndat"],
                    )
                else:
                    temp += " & & "
            temp += r" \\ \hline"
            L.append(temp)

        temp = r" \hline Total"
        for chi2_df_grouped in chi2_dict_group.values():
            temp += " & %d & %.3f (%.3f)" % (
                chi2_df_grouped.loc["Total", "ndat"],
                chi2_df_grouped.loc["Total", "chi2/ndat"],
                chi2_df_grouped.loc["Total", "chi2_sm/ndat"],
            )
        temp += r" \\ \hline"
        L.extend(
            [
                temp,
                r"\end{tabular}",
                r"\caption{$\chi^2$ table for grouped data. In parenthesis is the total SM $\chi^2$ for the dataset included in the fit.\
                    The SM column refers to all the datasets available in the group}",
                r"\end{table}",
            ]
        )
        return L

    def plot_exp(self, chi2_dict, fig_name):
        """Plots a bar plot of the chi2 values per experiment"""

        plt.figure(figsize=(10, 15))
        chi2_bar = pd.DataFrame()
        chi2_bar[r"${\rm SM}$"] = self.chi2_df_sm["chi2_sm/ndat"]
        for name, chi2_df in chi2_dict.items():
            chi2_bar[name] = chi2_df["chi2/ndat"]
        chi2_bar.index = [
            r"\rm{%s}" % name.replace("_", r"\_") for name in chi2_bar.index
        ]

        chi2_bar.plot(kind="barh", width=0.7, figsize=(10, 15))

        plt.plot(
            np.ones(2), np.linspace(-1, chi2_bar.shape[0] * 10, 2), "k--", alpha=0.7
        )
        plt.tick_params(axis="x", direction="in", labelsize=15)
        plt.xlabel(r"$\chi^2$", fontsize=20)
        plt.xlim(0, 4)
        plt.legend(loc="upper right", frameon=False, prop={"size": 11})
        plt.tight_layout()
        plt.savefig(fig_name)

    def plot_dist(self, chi2_hist, fig_name):
        """Plots chi2 distribution."""
        plt.figure(figsize=(7, 5))
        ax = plt.subplot(111)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i, (label, chi2_list) in enumerate(chi2_hist.items()):
            ax.hist(
                chi2_list,
                bins="fd",
                density=True,
                color=colors[i],
                edgecolor="k",
                alpha=0.3,
                label=label,
            )

        plt.tick_params(axis="x", direction="in", labelsize=15)
        plt.tick_params(axis="y", direction="in", labelsize=15)
        plt.xlabel(r"\rm $\chi^2$\ distribution", fontsize=20)
        plt.legend(loc=2, frameon=False, prop={"size": 11})
        plt.tight_layout()
        plt.savefig(fig_name)
