# -*- coding: utf-8 -*-
import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns
from matplotlib import patches, transforms
from matplotlib.patches import Ellipse

from .coefficients_utils import CoefficientsPlotter


class contour2dPlotter(CoefficientsPlotter):
    def __int__(self, is_pairwise, report_path, free_coeff_config):
        super().__init__(report_path, free_coeff_config)
        self.is_pairwise = is_pairwise

    def plot_contours_2d(self, posteriors, labels, confidence_level=95, dofs_show=None):
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

        if dofs_show is not None:
            posteriors = [
                (posterior[0][dofs_show], posterior[1]) for posterior in posteriors
            ]
            coeff = dofs_show
            n_par = len(dofs_show)
        else:
            coeff = self.coeff_df.index
            n_par = self.npar

        n_cols = n_par - 1
        n_rows = n_cols

        fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))

        grid = plt.GridSpec(n_rows, n_cols, hspace=0.1, wspace=0.1)

        c1_old = coeff[0]

        row_idx = -1
        col_idx = -1
        j = 1

        # loop over coefficient pairs
        for (c1, c2) in itertools.combinations(coeff, 2):

            if c1 != c1_old:
                row_idx += -1
                col_idx = -1 - j
                j += 1
                c1_old = c1

            ax = fig.add_subplot(grid[row_idx, col_idx])

            # loop over fits
            hndls_all = []
            for clr_idx, (posterior, kde) in enumerate(posteriors):

                hndls_contours = plot_contours(
                    ax,
                    posterior,
                    coeff1=c1,
                    coeff2=c2,
                    ax_labels=[
                        self.coeff_df["latex_name"][c1],
                        self.coeff_df["latex_name"][c2],
                    ],
                    kde=kde,
                    clr_idx=clr_idx,
                    confidence_level=confidence_level,
                )

                hndls_all.append(hndls_contours)

                if row_idx != -1:
                    ax.set(xlabel=None)
                    ax.tick_params(
                        axis="x",  # changes apply to the x-axis
                        which="both",  # both major and minor ticks are affected
                        labelbottom=False,
                    )
                if col_idx != -n_cols:
                    ax.set(ylabel=None)
                    ax.tick_params(
                        axis="y",  # changes apply to the y-axis
                        which="both",  # both major and minor ticks are affected
                        labelleft=False,
                    )

            hndls_sm_point = ax.scatter(0, 0, c="k", marker="+", s=50, zorder=10)
            hndls_all.append(hndls_sm_point)

            col_idx -= 1

        ax = fig.add_subplot(grid[0, 1])
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
            r"$\mathrm{Marginalised}\:95\:\%\:\mathrm{C.L.\:intervals}$", fontsize=24
        )
        grid.tight_layout(fig)
        fig.savefig(f"{self.report_folder}/contours_2d.pdf")
        fig.savefig(f"{self.report_folder}/contours_2d.png")


def confidence_ellipse(
    coeff1, coeff2, ax, facecolor="none", confidence_level=95, **kwargs
):
    """
    Draws 95% C.L. ellipse for data points `x` and `y`

    Parameters
    ----------
    coeff1: array_like
        ``(N,) ndarray`` containing ``N`` posterior samples for the first coefficient
    coeff2: array_like
        ``(N,) ndarray`` containing ``N`` posterior samples for the first coefficient
    ax: matplotlib.axes
        Axes object to plot on
    facecolor: str, optional
        Color of the ellipse
    **kwargs
        Additional plotting settings passed to matplotlib.patches.Ellipse

    Returns
    -------
    matplotlib.patches.Ellipse
        Ellipse object

    """

    if coeff1.size != coeff2.size:
        raise ValueError("coeff1 and coeff2 must be the same size")

    # construct covariance matrix of coefficients
    cov = np.cov(coeff1, coeff2)

    # diagonalise
    eig_val, eig_vec = np.linalg.eig(cov)

    # eigenvector with largest eigenvalue
    eig_vec_max = eig_vec[:, np.argmax(eig_val)]

    # angle of eigenvector with largest eigenvalue with the horizontal axis
    cos_th = eig_vec[0, np.argmax(eig_val)] / np.linalg.norm(eig_vec_max)
    if eig_vec_max[1] > 0:
        inclination = np.arccos(cos_th)
    else:
        # pay attention to range of arccos (extend to [0, -\pi] domain)
        inclination = -np.arccos(cos_th)

    eigval_sort = np.sort(eig_val)

    chi2_qnt = scipy.stats.chi2.ppf(confidence_level / 100.0, 2)

    ell_radius_x = np.sqrt(chi2_qnt * eigval_sort[-1])
    ell_radius_y = np.sqrt(chi2_qnt * eigval_sort[-2])

    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    mean_coeff1 = np.median(coeff1)
    mean_coeff2 = np.median(coeff2)

    transf = (
        transforms.Affine2D().rotate(inclination).translate(mean_coeff1, mean_coeff2)
    )

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


def plot_contours(
    ax, posterior, ax_labels, coeff1, coeff2, kde, clr_idx, confidence_level=95
):
    """

    Parameters
    ----------
    ax: matplotlib.axes
        Axes object to plot on
    posterior: pandas.DataFrame
        Posterior samples per coefficient
    ax_labels: list
        Latex names
    coeff1: str
        Name of first coefficient
    coeff2: str
        Name of second coefficient
    kde: bool
        Performs kernel density estimate (kde) when quadratics are turned on
    clr_idx: int
        Color index that makes sure each fit gets associated a different color
    confidence_level: int, optional
        Confidence level interval, set to 95% by default

    Returns
    -------
    hndls: list
        List of Patch objects
    """

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    x_values = posterior[coeff2].values
    y_values = posterior[coeff1].values

    # perform kde for quadratic EFT fit
    if kde:
        sns.kdeplot(
            x=x_values,
            y=y_values,
            levels=[1 - confidence_level / 100.0, 1.0],
            bw_adjust=1.2,
            ax=ax,
            fill=True,
            alpha=0.3,
            color=colors[clr_idx],
        )
        sns.kdeplot(
            x=x_values,
            y=y_values,
            levels=[1 - confidence_level / 100.0],
            bw_adjust=1.2,
            ax=ax,
            alpha=1,
            color=colors[clr_idx],
        )

        hndls = (
            patches.Patch(ec=colors[clr_idx], fc=colors[clr_idx], fill=True, alpha=0.3),
            patches.Patch(
                ec=colors[clr_idx], fc=colors[clr_idx], fill=False, alpha=1.0
            ),
        )

    else:  # draw ellipses for linear EFT fit

        p1 = confidence_ellipse(
            x_values,
            y_values,
            ax,
            alpha=1,
            edgecolor=colors[clr_idx],
            confidence_level=confidence_level,
        )

        p2 = confidence_ellipse(
            x_values,
            y_values,
            ax,
            alpha=0.3,
            facecolor=colors[clr_idx],
            edgecolor=None,
            confidence_level=confidence_level,
        )

        ax.scatter(
            np.mean(x_values, axis=0),
            np.mean(y_values, axis=0),
            c=colors[clr_idx],
            s=3,
        )

        hndls = (p1, p2)

    plt.xlabel(ax_labels[1], fontsize=26)
    plt.ylabel(ax_labels[0], fontsize=26)

    plt.tick_params(which="both", direction="in", labelsize=22)

    return hndls
