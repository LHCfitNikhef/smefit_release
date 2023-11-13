import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib import rc, use
import subprocess

from histogram_tools import find_xrange
from latex_dicts import mod_dict
from latex_dicts import uv_param_dict
from latex_dicts import inv_param_dict

use("PDF")
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", **{"usetex": True, "latex.preamble": r"\usepackage{amssymb}"})

pQCD = "NLO"

tree_NLO_lin = '/data/theorie/jthoeve/smefit_release/results/Granada_UV_5_NLO_NHO_NS/inv_posterior.json'
tree_NLO_quad = '/data/theorie/jthoeve/smefit_release/results/Granada_UV_5_NLO_HO_NS/inv_posterior.json'
loop_NLO_lin = '/data/theorie/jthoeve/smefit_release/results/OneLoop_UV_Varphi_NLO_NHO_NS/inv_posterior.json'
loop_NLO_quad = '/data/theorie/jthoeve/smefit_release/results/OneLoop_UV_Varphi_NLO_HO_NS/inv_posterior.json'


with open(tree_NLO_lin) as f:
    posterior_tree_lin = json.load(f)

with open(tree_NLO_quad) as f:
    posterior_tree_quad = json.load(f)

with open(loop_NLO_lin) as f:
    posterior_loop_lin = json.load(f)

with open(loop_NLO_quad) as f:
    posterior_loop_quad = json.load(f)


def draw_hist(ax, samples, threshold_x, label):
    x_low, x_high = find_xrange(samples, threshold_x)

    for sample in samples:
        ax.hist(
            sample[(sample > x_low) & (sample < x_high)],
            bins="fd",
            density=True,
            edgecolor="black",
            alpha=0.4,
        )

    ax.set_xlim(x_low, x_high)
    ax.set_ylim(0, ax.get_ylim()[1] + ax.get_ylim()[1] * 0.2)

    x_pos = ax.get_xlim()[0] + 0.05 * (
            ax.get_xlim()[1] - ax.get_xlim()[0]
    )
    y_pos = 0.95 * ax.get_ylim()[1]

    ax.tick_params(which="both", direction="in", labelsize=22.5)
    ax.tick_params(labelleft=False)

    ax.text(
        x_pos,
        y_pos,
        label[0],
        fontsize=15,
        ha="left",
        va="top",
    )

    ax.text(
        0.95,
        0.95,
        label[1],
        fontsize=15,
        ha="right",
        va="top",
        transform=ax.transAxes
    )

n_cols = 3
n_rows = 2
fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))


samples_1 = np.array(posterior_tree_lin['inv1'])
samples_2 = np.array(posterior_tree_quad['inv1'])
draw_hist(ax[0,0], [samples_1, samples_2], 0.05, [inv_param_dict[5]['inv1'], r"$\mathrm{Tree\:level}$"])


samples_1 = np.array(posterior_tree_lin['inv2'])
samples_2 = np.array(posterior_tree_quad['inv2'])
draw_hist(ax[0,1], [samples_1, samples_2], 0.05, [inv_param_dict[5]['inv2'], r"$\mathrm{Tree\:level}$"])

samples_1 = np.array(posterior_loop_lin['inv1'])
samples_2 = np.array(posterior_loop_quad['inv1'])
draw_hist(ax[1,0], [samples_1, samples_2], 0.02, [inv_param_dict[5]['inv1'], r"$\textrm{One-loop}$"])


samples_1 = np.array(posterior_loop_lin['inv2'])
samples_2 = np.array(posterior_loop_quad['inv2'])
draw_hist(ax[1,1], [samples_1, samples_2], 0.02, [inv_param_dict[5]['inv2'], r"$\textrm{One-loop}$"])

yVarphiuf33 = np.array(posterior_tree_lin['yVarphiuf33'])
lamVarphi = np.array(posterior_tree_lin['lamVarphi'])

ax[0,2].scatter(yVarphiuf33, lamVarphi, s=1, color='C0', alpha=0.4)
ax[0,2].set_xlim(-0.05, 0.05)
ax[0,2].set_xlabel(r'$\left(y_\varphi^u\right)_{33}$')
ax[0,2].set_ylabel(r'$\lambda_\varphi$')
ax[0,2].text(
        0.95,
        0.95,
        r"$\mathrm{Tree\:level}$",
        fontsize=15,
        ha="right",
        va="top",
        transform=ax[0,2].transAxes
    )

yVarphiuf33 = np.array(posterior_loop_lin['yVarphiuf33'])
lamVarphi = np.array(posterior_loop_lin['lamvarphi'])

ax[1,2].scatter(yVarphiuf33, lamVarphi, s=1, color='C0', alpha=0.4)
#ax[1,2].set_xlim(-0.05, 0.05)
ax[1,2].set_xlabel(r'$\left(y_\phi^u\right)_{33}$')
ax[1,2].set_ylabel(r'$\lambda_\phi$')
ax[1,2].text(
        0.95,
        0.95,
        r"$\textrm{One-loop}$",
        fontsize=15,
        ha="right",
        va="top",
        transform=ax[1,2].transAxes
    )
#ax[1,2].set_xlim(-0.05, 0.05)
# ax[0,2].set_xlabel(r'$\left|\left(y_\varphi^u\right)_{33}\right|$')
# ax[0,2].set_ylabel(r'$\mathrm{sgn}\left(y_\varphi^u\right)_{33}\lambda_\varphi$')



# samples_1 = np.array(posterior_tree_lin['yVarphiuf33']) * np.array(posterior_tree_lin['lamVarphi'])
# samples_2 = np.array(posterior_tree_quad['yVarphiuf33']) * np.array(posterior_tree_quad['lamVarphi'])
# draw_hist(ax[0,3], [samples_1, samples_2], 0.02, "$y\lambda$")
#
# samples_1 = np.array(posterior_loop_lin['yVarphiuf33']) * np.array(posterior_loop_lin['lamvarphi'])
# samples_2 = np.array(posterior_loop_quad['yVarphiuf33']) * np.array(posterior_loop_quad['lamvarphi'])
# draw_hist(ax[1,3], [samples_1, samples_2], 0.02, "$y\lambda$")


plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # make room for the legend
fig.legend([f"$\mathrm{{{pQCD}}}\;\mathcal{{O}}\\left(\Lambda^{{-2}}\\right)$",
                    f"$\mathrm{{{pQCD}}}\;\mathcal{{O}}\\left(\Lambda^{{-4}}\\right)$"], ncol=2,
                   prop={"size": 25 * (n_cols * 4) / 20}, bbox_to_anchor=(0.5, 1.0), loc='upper center', frameon=False)

fig.savefig('./tree_vs_loop.pdf')
subprocess.run(["pdfcrop", './tree_vs_loop.pdf', './tree_vs_loop.pdf'])