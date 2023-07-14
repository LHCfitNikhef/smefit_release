import matplotlib.pyplot as plt
import json
from pathlib import Path
from matplotlib import rc, use
import numpy as np
import itertools
import pandas as pd
from matplotlib.gridspec import GridSpec
from latex_dicts import inv_param_dict

use("PDF")
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", **{"usetex": True, "latex.preamble": r"\usepackage{amssymb}"})

# Specify the path to the JSON file
posterior_path_lo = "/data/theorie/jthoeve/smefit_release/results/Granada_UV_25_LO_NHO_NS/inv_posterior.json"

with open(posterior_path_lo) as f:
    posterior_inv = json.load(f)

df = pd.DataFrame.from_dict(posterior_inv)
df_inv = df.filter(like="inv", axis=1)
n_invariants = df_inv.shape[-1]


n_rows = n_invariants - 1
n_cols = n_rows
fig = plt.figure(figsize=(7, 7))
grid = GridSpec(n_rows, n_cols, height_ratios=[1, 1], hspace=0.1, wspace=0.1)

row_idx = -1
col_idx = -1
j = 1
inv_old = "inv1"

for (inv1, inv2) in itertools.combinations(df_inv.columns, 2):

    print(inv1, inv2)
    if inv1 != inv_old:
        row_idx += -1
        col_idx = -1 - j
        j += 1
        inv_old = inv1

    ax = fig.add_subplot(grid[row_idx, col_idx])
    ax.scatter(df_inv[inv2], df_inv[inv1], s=1)
    ax.set_xlabel(inv_param_dict[25][inv2])
    ax.set_ylabel(inv_param_dict[25][inv1])
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

fig.savefig('/data/theorie/jthoeve/smefit_release/smefit_uv_test/inv_marg.pdf')