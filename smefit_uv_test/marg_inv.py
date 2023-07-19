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

#use("PDF")
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], 'size': 17})
rc("text", **{"usetex": True, "latex.preamble": r"\usepackage{amssymb}"})

# Specify the path to the JSON file
posterior_path_lin = "/data/theorie/jthoeve/smefit_release/results/MP_UV_Q1_Q7_W1_LO_NHO_NS/inv_posterior.json"
posterior_path_quad = "/data/theorie/jthoeve/smefit_release/results/MP_UV_Q1Q7W1_ND_LO_NHO_NS/inv_posterior.json"

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
samples = random.sample(range(tot_samples), 10)


def cornerplot(dfs, tex_dict, path, labels):

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    hndls = []
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

        for i, df in enumerate(dfs):
            #ax.scatter(df[inv2].iloc[samples], df[inv1].iloc[samples], s=1)
            sns.kdeplot(x=df[inv2].iloc[samples], y=df[inv1].iloc[samples], levels=[0.05, 1.0], bw_adjust=1.8, ax=ax,
                       fill=True, alpha=0.4)

            sns.kdeplot(x=df[inv2].iloc[samples], y=df[inv1].iloc[samples], levels=[0.05], bw_adjust=1.8, ax=ax, alpha=1)
            hndls.append((mpatches.Patch(ec=colors[i], fc=colors[i], fill=True, alpha=0.4),
                          mpatches.Patch(ec=colors[i], fc=colors[i], fill=False, alpha=1.0)))

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
        #plt.tight_layout(rect=[0, 0, 1, 1])
    #
    # legend = ax.legend(
    #     labels=labels, handles=hndls,
    #     bbox_to_anchor=(1, 1),
    #     loc='upper left', frameon=False, fontsize=24,
    #     handlelength=1,
    #     borderpad=0.5,
    #     handletextpad=1,
    #     title_fontsize=24)

    fig.legend(labels=labels, handles=hndls,
               prop={"size": 25 * (n_cols * 4) / 20}, bbox_to_anchor=(0.5, 1 - .5/n_rows), loc='upper center', frameon=False)

    #bbox = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    grid.tight_layout(fig, rect=[0, 0.05 * (5. / n_rows), 1, 1 - 0.05 * (5. / n_rows)])

    fig.savefig('{}.pdf'.format(path))


dfs_inv = [df_inv_lin, df_inv_quad]
dfs_uv_params = [df_uv_params_lin, df_uv_params_quad]
labels = [r"$m_{Q_1}=m_{Q_7}=m_{\mathcal{W}_1}=1.0\:\mathrm{TeV}$",
            r"$m_{Q_1}=3.0\:\mathrm{TeV}, m_{Q_7}=4.5\:\mathrm{TeV}, m_{\mathcal{W}_1}=2.5\:\mathrm{TeV}$"]

# cornerplot(dfs_inv, inv_param_dict["Q1_Q7_W1"],
#           '/data/theorie/jthoeve/smefit_release/smefit_uv_test/marg_inv_plots/inv_multiparticle_nd_marg', labels)

cornerplot(dfs_uv_params, uv_param_dict, '/data/theorie/jthoeve/smefit_release/smefit_uv_test/marg_inv_plots/uv_multiparticle_nd_marg_v2',
           labels)
