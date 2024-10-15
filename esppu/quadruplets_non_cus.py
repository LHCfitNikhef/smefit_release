# -*- coding: utf-8 -*-
import importlib
import json
import pathlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 22})
rc("text", usetex=True)

collections = ["Granada"]


here = pathlib.Path("/Users/jaco/Documents/smefit_release/results/esppu")

# result dir
# result_dir = here / "FCCee-HLLHC-Custo-EWquad"
result_dir = here / "FCCee-HLLHC-Quad1-EW"

ncols, nrows = 4, 4
fig, ax = plt.subplots(figsize=(4 * ncols, 4 * nrows))


def get_bounds():

    bound_dict = {}
    i = 0
    for path in pathlib.Path(result_dir).iterdir():
        posterior_path = path / "posterior.json"
        if posterior_path.exists():

            # compute bounds
            with open(posterior_path, "r") as f:
                posterior = json.load(f)
                posterior_uv = np.array(posterior["lamPhi"])
                bound_up = np.percentile(np.abs(posterior_uv), 95)
                bound_dict[path.stem] = bound_up

            ax = plt.subplot(ncols, nrows, i + 1)
            # plot posterior
            ax.hist(
                posterior_uv,
                bins="fd",
                density=True,
                edgecolor="black",
                alpha=0.4,
            )

            ax.set_title(path.stem, size=10)
            i += 1

            plt.tight_layout()
            plt.savefig("./results/posterior_non_custodial.pdf")

    return bound_dict


bounds_uv = get_bounds()

bounds_uv_fcc_hllhc = {}
for key, val in bounds_uv.items():
    if key == "HLLHC_UV_EW_Quad1_tree_4_NLO_HO_NS_noHH":
        continue
    if key.startswith("HLLHC"):
        suffix = key.split("_", 1)[1]
        bounds_uv_fcc_hllhc[suffix] = [val, bounds_uv["FCCee_" + suffix]]


df_uv = pd.DataFrame.from_dict(
    bounds_uv_fcc_hllhc, columns=["FCCee", "HLLHC"], orient="index"
)

df_uv = df_uv.reindex(
    index=[
        "UV_EW_Quad1_tree_4_NLO_HO_NS_norge",
        "UV_EW_Quad1_tree_4_NLO_HO_NS",
        "UV_EW_Quad1_1loop_4_NLO_HO_NS_norgenoHH",
        "UV_EW_Quad1_1loop_4_NLO_HO_NS_noHH",
        "UV_EW_Quad1_1loop_4_NLO_HO_NS_norge",
        "UV_EW_Quad1_1loop_4_NLO_HO_NS",
    ]
)


x_labels = [
    r"$\mathrm{tree,\:no\:RG,\:HH}$",
    r"$\mathrm{tree,\:RG,\:HH}$",
    r"$\mathrm{1L,\:no\:RG,\:no\:HH}$",
    r"$\mathrm{1L,\:RG,\:no\:HH}$",
    r"$\mathrm{1L,\:no\:RG,\:HH}$",
    r"$\mathrm{1L,\:RG,\:HH}$",
]
df_uv["xlabel"] = x_labels
# drop flat direction: no sensitivity to H6 with diHiggs data removed
# bounds_uv = bounds_uv.drop("FCCee_UV_EW_Quad13_Custo_tree_4_NLO_HO_NS_noHH")

# hllhc_index = bounds_uv.index.str.contains('HLLHC')
# fcc_index = bounds_uv.index.str.contains('FCCee')
#
# bounds_hllhc = bounds_uv[hllhc_index]
# bounds_fcc = bounds_uv[fcc_index]


# for bound_fcc in bounds_fcc.index:
#     import pdb; pdb.set_trace()
#     suffix = bound_fcc.split("FCCee")[1]
#     if "HLLHC" + suffix in bounds_hl
# bounds_hllhc = bounds_uv[hllhc_index]


fig, ax = plt.subplots(figsize=(17, 13))
df_uv.plot(kind="bar", ax=ax, x="xlabel", xlabel="", rot=45)
ax.legend(
    [r"$\rm{LEP+LHC_{Run-2}+\:HL-LHC}$", r"$\rm{LEP+LHC_{Run-2}+\:HL-LHC+\:FCC-ee}$"],
    frameon=False,
)
ax.set_ylabel(r"$|\lambda_\phi|$")
ax.set_title(
    r"$\mathrm{95\:\%\:CL\:bounds,\:NLO\:}\mathcal{O}\left(\Lambda^{-4}\right),\:\mathrm{non\textnormal{-}custodial}$",
    y=1.05,
)
plt.tight_layout()

# plt.legend(loc='upper right')
plt.savefig("./results/barplot_quadruplet_non_custodial.pdf")
