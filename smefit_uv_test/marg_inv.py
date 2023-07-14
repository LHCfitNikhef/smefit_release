import matplotlib.pyplot as plt
import json
from pathlib import Path
from matplotlib import rc, use
import numpy as np
import itertools
import pandas as pd
import random
from matplotlib.gridspec import GridSpec
from latex_dicts import inv_param_dict
from latex_dicts import uv_param_dict

use("PDF")
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], 'size': 17})
rc("text", **{"usetex": True, "latex.preamble": r"\usepackage{amssymb}"})

# Specify the path to the JSON file
posterior_path_lo = "/data/theorie/jthoeve/smefit_release/results/Granada_UV_25_LO_NHO_NS/inv_posterior.json"


with open(posterior_path_lo) as f:
    posterior_inv = json.load(f)

df = pd.DataFrame.from_dict(posterior_inv)
df_inv = df.filter(like="inv", axis=1)
n_invariants = df_inv.shape[-1]

uv_params = [name for name in df.columns if not name.startswith('O') and not name.startswith('inv')]
df_uv_params = df[uv_params]

tot_samples = df.shape[0]
samples = random.sample(range(tot_samples), 1000)

def cornerplot(df, tex_dict, path):

    n_params = df.shape[-1]
    n_rows = n_params - 1
    n_cols = n_rows

    fig = plt.figure(figsize=(4 * n_rows, 4 * n_cols))
    grid = GridSpec(n_rows, n_cols, hspace=0.1, wspace=0.1)
    
    row_idx = -1
    col_idx = -1
    j = 1
    inv_old = df.columns[0]
    
    for (inv1, inv2) in itertools.combinations(df.columns, 2):
    
        print(inv1, inv2)
        if inv1 != inv_old:
            row_idx += -1
            col_idx = -1 - j
            j += 1
            inv_old = inv1
    
        ax = fig.add_subplot(grid[row_idx, col_idx])

        ax.scatter(df[inv2].iloc[samples], df[inv1].iloc[samples], s=1)
        ax.set_xlabel(tex_dict[inv2])
        ax.set_ylabel(tex_dict[inv1])
        ax.tick_params(axis='both', direction='in', which='both')
    
        if row_idx != -1:
            ax.set(xlabel=None)
            ax.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                labelbottom=False)
        if col_idx != -n_cols:
            ax.set(ylabel=None)
            ax.tick_params(
                axis='y',  # changes apply to the y-axis
                which='both',  # both major and minor ticks are affected
                labelleft=False)
    
        col_idx -= 1

    fig.savefig('{}.pdf'.format(path))


cornerplot(df_inv, inv_param_dict[25], '/data/theorie/jthoeve/smefit_release/smefit_uv_test/marg_inv_plots/inv_marg_G')
cornerplot(df_uv_params, uv_param_dict, '/data/theorie/jthoeve/smefit_release/smefit_uv_test/marg_inv_plots/uv_marg_G')