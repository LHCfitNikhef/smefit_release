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
result_dir_cus = here / "FCCee-HLLHC-Custo-EWquad"
result_dir_non_cus = here / "FCCee-HLLHC-Quad1-EW"

ncols, nrows = 4, 4
fig, ax = plt.subplots(figsize=(4 * ncols, 4 * nrows))


def get_bounds(result_dir):

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

            # ax = plt.subplot(ncols, nrows, i + 1)
            # # plot posterior
            # ax.hist(
            #     posterior_uv,
            #     bins="fd",
            #     density=True,
            #     edgecolor="black",
            #     alpha=0.4,
            # )
            #
            # ax.set_title(path.stem, size=10)
            # i += 1
            #
            # plt.tight_layout()
            # plt.savefig('./results/posterior_non_custodial.pdf')

    return bound_dict


bounds_uv_non_cus = get_bounds(result_dir_non_cus)
bounds_uv_cus = get_bounds(result_dir_cus)

bounds_uv_non_cus_fcc_hllhc = {}
for key, val in bounds_uv_non_cus.items():
    if key == "HLLHC_UV_EW_Quad1_tree_4_NLO_HO_NS_noHH":
        continue
    if key.startswith("HLLHC"):
        suffix = key.split("_", 1)[1]
        bounds_uv_non_cus_fcc_hllhc[suffix] = [
            val,
            bounds_uv_non_cus["FCCee_" + suffix],
        ]

bounds_uv_cus_fcc_hllhc = {}
for key, val in bounds_uv_cus.items():
    if key == "HLLHC_UV_EW_Quad13_Custo_tree_4_NLO_HO_NS_noHH":
        continue
    if key.startswith("HLLHC"):
        suffix = key.split("_", 1)[1]
        bounds_uv_cus_fcc_hllhc[suffix] = [val, bounds_uv_cus["FCCee_" + suffix]]


df_uv_non_cus = pd.DataFrame.from_dict(
    bounds_uv_non_cus_fcc_hllhc, columns=["FCCee", "HLLHC"], orient="index"
)

df_uv_cus = pd.DataFrame.from_dict(
    bounds_uv_cus_fcc_hllhc, columns=["FCCee", "HLLHC"], orient="index"
)


df_uv_non_cus = df_uv_non_cus.reindex(
    index=[
        "UV_EW_Quad1_tree_4_NLO_HO_NS_norge",
        "UV_EW_Quad1_tree_4_NLO_HO_NS",
        "UV_EW_Quad1_1loop_4_NLO_HO_NS_norgenoHH",
        "UV_EW_Quad1_1loop_4_NLO_HO_NS_noHH",
        "UV_EW_Quad1_1loop_4_NLO_HO_NS_norge",
        "UV_EW_Quad1_1loop_4_NLO_HO_NS",
    ]
)

df_uv_cus = df_uv_cus.reindex(
    index=[
        "UV_EW_Quad13_Custo_tree_4_NLO_HO_NS_norge",
        "UV_EW_Quad13_Custo_tree_4_NLO_HO_NS",
        "UV_EW_Quad13_Custo_1loop_4_NLO_HO_NS_norgenoHH",
        "UV_EW_Quad13_Custo_1loop_4_NLO_HO_NS_noHH",
        "UV_EW_Quad13_Custo_1loop_4_NLO_HO_NS_norge",
        "UV_EW_Quad13_Custo_1loop_4_NLO_HO_NS",
    ]
)

df_uv = pd.concat([df_uv_cus, df_uv_non_cus])


x_labels = [
    r"$\mathrm{tree,\:no\:RG,\:HH}$",
    r"$\mathrm{tree,\:RG,\:HH}$",
    r"$\mathrm{1L,\:no\:RG,\:no\:HH}$",
    r"$\mathrm{1L,\:RG,\:no\:HH}$",
    r"$\mathrm{1L,\:no\:RG,\:HH}$",
    r"$\mathrm{1L,\:RG,\:HH}$",
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


fig, ax = plt.subplots(figsize=(20, 13))
df_uv.plot(kind="bar", ax=ax, x="xlabel", xlabel="", rot=45)
# ax.legend([r'$\rm{LEP+LHC_{Run-2}+\:HL-LHC}$', r"$\rm{LEP+LHC_{Run-2}+\:HL-LHC+\:FCC-ee}$"], frameon=False)
ax.set_ylabel(r"$|\lambda_\phi|$")
fig.legend(
    [r"$\rm{LEP+LHC_{Run-2}+\:HL-LHC}$", r"$\rm{LEP+LHC_{Run-2}+\:HL-LHC+\:FCC-ee}$"],
    loc="upper center",
    ncol=2,
    prop={"size": 25},
    bbox_to_anchor=(0.5, 1.0),
    frameon=False,
)
ax.axvline(5.5, 0, 1, color="k", ls="--")

legend = ax.legend()
legend.remove()

ax.text(
    0.02,
    0.95,
    r"$\mathrm{Custodial}$",
    horizontalalignment="left",
    verticalalignment="center",
    transform=ax.transAxes,
)

ax.text(
    0.98,
    0.95,
    r"$\mathrm{Non-custodial}$",
    horizontalalignment="right",
    verticalalignment="center",
    transform=ax.transAxes,
)

# ax.set_title(r"$\mathrm{95\:\%\:CL\:bounds,\:NLO\:}\mathcal{O}\left(\Lambda^{-4}\right)$", y=1.05)
plt.tight_layout(rect=[0, 0.05, 1, 1 - 0.05])

# plt.legend(loc='upper right')
plt.savefig("./results/barplot_quadruplet_all.pdf")
