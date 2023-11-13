import matplotlib.pyplot as plt
import json
from pathlib import Path
from matplotlib import rc, use
import numpy as np
import itertools
import pandas as pd
import random
import seaborn as sns
from matplotlib.gridspec import GridSpec
from latex_dicts import inv_param_dict
from latex_dicts import uv_param_dict
import matplotlib.patches as mpatches
import arviz as az

#use("PDF")
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], 'size': 17})
rc("text", **{"usetex": True, "latex.preamble": r"\usepackage{amssymb}"})

# Specify the path to the JSON file
posterior_path_lin = "/data/theorie/jthoeve/smefit_release/results/Multiparticle_UV_Q1_Q7_W_NLO_HO_NS/inv_posterior.json"
posterior_path_quad = "/data/theorie/jthoeve/smefit_release/results/Multiparticle_UV_Q1_Q7_W_NoDegen_NLO_HO_NS/inv_posterior.json"

# posterior_path_lin = "/data/theorie/jthoeve/smefit_release/results/Multiparticle_UV_Q1_Q7_W_NLO_HO_NS/posterior.json"
# posterior_path_quad = "/data/theorie/jthoeve/smefit_release/results/Multiparticle_UV_Q1_Q7_W_NoDegen_NLO_HO_NS/posterior.json"

with open(posterior_path_lin) as f:
    posterior_lin = json.load(f)

with open(posterior_path_quad) as f:
    posterior_quad = json.load(f)

df_lin = pd.DataFrame.from_dict(posterior_lin)
df_inv_lin = df_lin.filter(like="inv", axis=1)
df_quad = pd.DataFrame.from_dict(posterior_quad)
df_inv_quad = df_quad.filter(like="inv", axis=1)

n_invariants = df_inv_lin.shape[-1]

uv_params = [name for name in df_lin.columns if not name.startswith('O') and not name.startswith('inv')]
df_uv_params_lin = df_lin[uv_params]
df_uv_params_quad = df_quad[uv_params]

tot_samples = min(df_lin.shape[0], df_quad.shape[0])


samples = random.sample(range(tot_samples), 100)


def cornerplot(dfs, tex_dict, path, labels):

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    hndls = []

    #dfs[0] = dfs[0].drop('gWLf33', axis=1)

    dfs[0] = dfs[0].drop('inv3', axis=1)
    n_params = dfs[0].shape[-1]
    n_rows = n_params - 1
    n_cols = n_rows

    fig = plt.figure(figsize=(4 * n_rows, 4 * n_cols))
    grid = GridSpec(n_rows, n_cols)

    row_idx = -1
    col_idx = -1
    j = 1
    inv_old = dfs[0].columns[0]

    for (inv1, inv2) in itertools.combinations(dfs[0].columns, 2):

        print(inv1, inv2)
        if inv1 != inv_old:
            row_idx += -1
            col_idx = -1 - j
            j += 1
            inv_old = inv1

        ax = fig.add_subplot(grid[row_idx, col_idx])

        xmin_all = []
        xmax_all = []
        ymin_all = []
        ymax_all = []
        for i, df in enumerate(dfs):
            print(i)
            # ax.scatter(df[inv2].iloc[samples], df[inv1].iloc[samples], s=1, color=colors[i])
            sns.kdeplot(x=df[inv2], y=df[inv1], levels=[0.05, 1.0], bw_adjust=1.9, ax=ax,
                       fill=True, alpha=0.4)

            sns.kdeplot(x=df[inv2], y=df[inv1], levels=[0.05], bw_adjust=1.9, ax=ax, alpha=1)
            hndls.append((mpatches.Patch(ec=colors[i], fc=colors[i], fill=True, alpha=0.4),
                          mpatches.Patch(ec=colors[i], fc=colors[i], fill=False, alpha=1.0)))

            xmin, xmax = az.hdi(df[inv2].values, hdi_prob=.95)
            ymin, ymax = az.hdi(df[inv1].values, hdi_prob=.95)
            ymin_all.append(ymin)
            ymax_all.append(ymax)
            xmin_all.append(xmin)
            xmax_all.append(xmax)

        deltax = 1.1 * (max(xmax_all) - min(xmin_all))
        deltay = 1.1 * (max(ymax_all) - min(ymin_all))
        # ax.axvline(min(xmin_all) - deltax, color='k')
        # ax.axvline(max(xmax_all) + deltax, color='k')
        # ax.axhline(min(ymin_all) - deltay, color='k')
        # ax.axhline(max(ymax_all) + deltay, color='k')

        if min(xmin_all) >= 0:
            ax.set_xlim(0, min(1000, max(xmax_all) + deltax))
        else:
            ax.set_xlim(max(-1000, min(xmin_all) - deltax), min(1000, max(xmax_all) + deltax))

        if min(ymin_all) >= 0:
            ax.set_ylim(0, min(1000, max(ymax_all) + deltay))
        else:
            ax.set_ylim((max(-1000, min(ymin_all) - deltay), min(1000, max(ymax_all) + deltay)))

        ax.set_xlabel(tex_dict[inv2], fontsize = 20 * (n_cols * 4) / 20)
        ax.set_ylabel(tex_dict[inv1], fontsize = 20 * (n_cols * 4) / 20)
        ax.tick_params(axis='both', direction='in', which='both', labelsize=20 * (n_cols * 4) / 20)


        if row_idx != -1:
            ax.set(xlabel=None)
            ax.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                labelbottom=False,
            )
        if col_idx != -n_cols:
            ax.set(ylabel=None)
            ax.tick_params(
                axis='y',  # changes apply to the y-axis
                which='both',  # both major and minor ticks are affected
                labelleft=False)

        col_idx -= 1


    fig.legend(labels=labels, handles=hndls,
               prop={"size": 25 * (n_cols * 4) / 20}, bbox_to_anchor=(0.5, 1 - .5/n_rows), loc='upper center', frameon=False)


    grid.tight_layout(fig, rect=[0, 0.05 * (5. / n_rows), 1, 1 - 0.05 * (5. / n_rows)])

    fig.suptitle(
        r"$(Q_1, Q_7, \mathcal{W})\mathrm{\;model,}\;95\:\%\:\mathrm{C.L.\:intervals\:at}\;\mathcal{O}\left(\Lambda^{-4}\right)$",
        y=0.93,
    fontsize=28)

    fig.savefig('{}.pdf'.format(path))


dfs_inv = [df_inv_lin, df_inv_quad]
dfs_uv_params = [df_uv_params_lin, df_uv_params_quad]
labels = [r"$m_{Q_1}=m_{Q_7}=m_{\mathcal{W}}=1.0\:\mathrm{TeV}$",
            r"$m_{Q_1}=3.0\:\mathrm{TeV}, m_{Q_7}=4.5\:\mathrm{TeV}, m_{\mathcal{W}}=2.5\:\mathrm{TeV}$"]

# cornerplot(dfs_inv, inv_param_dict["Q1_Q7_W1"],
#           '/data/theorie/jthoeve/smefit_release/smefit_uv_test/marg_inv_plots/inv_multiparticle_nd_marg', labels)

# cornerplot(dfs_uv_params, uv_param_dict, '/data/theorie/jthoeve/smefit_release/smefit_uv_test/marg_inv_plots/uv_multiparticle_nd_marg_final_95',
#            labels)

cornerplot(dfs_inv, inv_param_dict["Q1_Q7_W"], '/data/theorie/jthoeve/smefit_release/smefit_uv_test/marg_inv_plots/uv_multiparticle_inv_95_mass_comp',
           labels)
