# -*- coding: utf-8 -*-
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..compute_theory import flatten
from .latex_tools import latex_packages


def make_sym_matrix(vals, n_op):
    """Build a square tensor (n_op,n_op,vals.shape[0]), starting from the upper tiangular part.

    Parameters
    ----------
        vals : np.ndarray
            traingular part
        n_op : int
            dimension of the final matrix

    Returns
    -------
    np.ndarry:
        square tensor.

        ````
        make_sym_matrix(array([1,2,3,4,5,6]), 3) -> array([[1,2,3],[0,4,5],[0,0,6]])
        ````
    """
    n_dat = vals.shape[0]
    m = np.zeros((n_op, n_op, n_dat))
    xs, ys = np.triu_indices(n_op)
    for i, l in enumerate(vals):
        m[xs, ys, i] = l
        m[ys, xs, i] = l
    return m


def impose_constrain(dataset, coefficients, update_quad=False):
    """Propagate coefficient constraint into the theory tables.

    Note: only linear contraints are allowed in this method.
    Non linear contrains not always make sense here.

    Parameters
    ----------
    dataset: smefit.loader.DataTuple
        loaded datasets
    coefficient: smefit.coefficients.CoefficienManager
        coefficient manager
    update_quad: bool, optional
        if True update also quadratic corrections

    Returns
    -------
    np.ndarray
        array of updated linear corrections (n_free_op, n_dat)
    np.ndarray, optional
        array of updated quadratic corrections (n_free_op, n_free_op, n_dat)

    """
    temp_coeffs = copy.deepcopy(coefficients)
    free_coeffs = temp_coeffs.free_parameters.index
    n_free_params = free_coeffs.size
    new_linear_corrections = []
    new_quad_corrections = []
    # loop on the free op and add the corrections
    for idx in range(n_free_params):
        # update all the free coefficents to 0 except from 1 and propagate
        params = np.zeros_like(free_coeffs)
        params[idx] = 1.0
        temp_coeffs.set_free_parameters(params)
        temp_coeffs.set_constraints()

        # update linear corrections
        new_linear_corrections.append(temp_coeffs.value @ dataset.LinearCorrections.T)

        # update quadratic corrections, this will include some double counting in the mixed corrections
        if update_quad:
            for jdx in range(free_coeffs[idx:].size):
                params = np.zeros_like(free_coeffs)
                params[idx + jdx] = 1.0
                params[idx] = 1.0
                temp_coeffs.set_free_parameters(params)
                temp_coeffs.set_constraints()
                coeff_outer_coeff = np.outer(temp_coeffs.value, temp_coeffs.value)
                new_quad_corrections.append(
                    flatten(coeff_outer_coeff) @ dataset.QuadraticCorrections.T
                )

    if update_quad:
        # subrtact the squuared corrections from the mixed ones
        new_quad_corrections = make_sym_matrix(
            np.array(new_quad_corrections).T, n_free_params
        )
        for idx in range(n_free_params):
            for jdx in range(n_free_params):
                if jdx != idx:
                    new_quad_corrections[idx, jdx, :] -= (
                        new_quad_corrections[idx, idx, :]
                        + new_quad_corrections[jdx, jdx, :]
                    )
        return np.array(new_linear_corrections), new_quad_corrections

    return np.array(new_linear_corrections)


class PcaCalculator:
    """Computes and writes PCA table and heat map.

    Note: matrix being decomposed by SVD are the
    linear corrections multiplied by the inverse covariance matrix.

    Parameters
    ----------
    dataset: smefit.loader.DataTuple
        loaded datasets
    coefficients:  smefit.coefficients.CoefficienManager
        coefficient manager
    latex_names: dict
        coefficient latex names

    """

    def __init__(self, datasets, coefficients, latex_names):

        self.coefficients = coefficients
        self.datasets = datasets
        self.latex_names = latex_names
        self.pc_matrix = None
        self.SVs = None

    def compute(self):
        """Compute PCA."""
        free_parameters = self.coefficients.free_parameters.index

        new_LinearCorrections = impose_constrain(self.datasets, self.coefficients)
        X = new_LinearCorrections @ self.datasets.InvCovMat @ new_LinearCorrections.T
        # Decompose matrix with SVD and identify PCs
        Vt, W, _ = np.linalg.svd(X)

        pca_labels = [f"PC {i+1}" for i in range(W.size)]
        self.pc_matrix = pd.DataFrame(Vt, index=free_parameters, columns=pca_labels)
        self.SVs = pd.Series(W, index=pca_labels)

    def write(self, fit_label, thr_show=1e-2):
        """Write PCA latex table.

        Parameters
        ----------
        fit_label: str
            fit label
        thr_show: float
            minimal threshold to show in the PCA decomposition
        """
        L = latex_packages()
        L.extend(
            [
                r"\usepackage{underscore}",
                r"\allowdisplaybreaks",
                r"\renewcommand{\baselinestretch}{1.5}",
                r"\begin{document}",
                r"\noindent \underline{\bf{Principal Components Analysis}:} "
                + fit_label
                + r"\\ \\ \\",
            ]
        )
        # PCA Table, loop on PC
        for sv_name, sv_value in self.SVs.items():
            L.append(
                f"\\noindent \\textcolor{{red}}{{\\underline{{\\bf{{{sv_name}}} ({sv_value:.2e}):}}}}"
            )
            # loop on PC entries
            pc_sorted = self.pc_matrix[sv_name].sort_values(ascending=False, key=np.abs)
            for coeff, aij in pc_sorted[np.abs(pc_sorted) > thr_show].items():
                L.append(f"{{${aij:+0.3f}$}}{{\\rm {self.latex_names[coeff]}}} ")
            L.append(r" \nonumber \\ \nonumber \\ ")
        return L

    def plot_heatmap(
        self,
        fit_label,
        fig_name,
        sv_min=1e-4,
        sv_max=1e5,
        thr_show=0.1,
        figsize=(15, 15),
    ):
        """Heat Map of PC coefficients.

        Parameters
        ----------
        fit_label: str
            fit label
        fig_name: str
            plot name
        sv_min: float
            minimum singular value range shown in the top heatmap plot
        sv_max: float
            maximum singular value range shown in the top heatmap plot
        thr_show: float
            minimal threshold to show in the PCA decomposition
        """

        pc_norm = self.pc_matrix.values**2

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        cmap = cm.get_cmap("Blues")
        norm = colors.BoundaryNorm(np.arange(1.1, step=0.1), cmap.N)

        cax = ax.matshow(pc_norm, cmap=cmap, norm=norm)
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(cax, cax=cax1)

        cbar.set_label(r"${\rm a}_i^2$", fontsize=25, labelpad=30, rotation=270)
        cbar.ax.tick_params(labelsize=15)

        for i, row in enumerate(pc_norm):
            for j, pc in enumerate(row):
                if pc > thr_show:
                    ax.text(
                        j,
                        i,
                        f"{pc:.1f}",
                        va="center",
                        ha="center",
                        fontsize=10,
                    )

        # major grid
        ticks = np.arange(pc_norm.shape[0])
        ax.set_yticks(ticks, labels=self.latex_names[self.pc_matrix.index], fontsize=15)
        ax.set_xticks(
            ticks,
            labels=[f"\\rm {sv}" for sv in self.pc_matrix.columns],
            rotation=90,
            fontsize=15,
        )
        ax.tick_params(
            which="major", top=False, labelbottom=True, bottom=False, left=False
        )

        # minor grid
        ax.set_xticks(ticks - 0.5, minor=True)
        ax.set_yticks(ticks - 0.5, minor=True)
        ax.tick_params(which="minor", bottom=True)
        ax.grid(visible=True, which="minor", alpha=0.2)

        # Bar Plot of Singular Values
        ax_sv = divider.append_axes("top", size="40%", pad=0.1)
        ax_sv.bar(ticks, self.SVs.values, align="center")
        ax_sv.tick_params(which="major", labelbottom=False, bottom=False)
        ax_sv.set_xticks(ticks - 0.5, minor=True)
        ax_sv.tick_params(which="minor", bottom=True)
        ax_sv.set_yscale("log")
        ax_sv.set_ylim(sv_min, sv_max)
        ax_sv.set_xlim(-0.5, ticks.size - 0.5)
        ax_sv.set_ylabel(r"${\rm Singular\ Values}$", fontsize=20)

        # save
        ax.set_title(f"\\rm PCA:\\ {fit_label}", fontsize=25, y=-0.15)
        plt.tight_layout()
        plt.savefig(f"{fig_name}.pdf")
        plt.savefig(f"{fig_name}.png")
