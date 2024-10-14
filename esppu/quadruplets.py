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
result_dir = here / "FCCee-HLLHC-Custo-EWquad"

# ncols, nrows = 4, 4
# fig, ax = plt.subplots(figsize=(4 * ncols, 4 * nrows))


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
            # plt.savefig('./results/posterior.pdf')

    return bound_dict


bounds_uv = get_bounds()

bounds_uv_fcc_hllhc = {}
for key, val in bounds_uv.items():
    if key == "FCCee_UV_EW_Quad13_Custo_tree_4_NLO_HO_NS_noHH":
        continue
    if key.startswith("FCCee"):
        suffix = key.split("_", 1)[1]

        bounds_uv_fcc_hllhc[suffix] = [val, bounds_uv["HLLHC_" + suffix]]


df_uv = pd.DataFrame.from_dict(
    bounds_uv_fcc_hllhc, columns=["FCCee", "HLLHC"], orient="index"
)

x_labels = [
    "1-loop, No RG, No HH",
    "tree, RG, HH",
    "1-loop, RG, No HH",
    "tree, No RG, HH",
    "1-loop, No RG, HH",
    "1-loop, RG, HH",
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


fig, ax = plt.subplots(figsize=(13, 13))
df_uv.plot(kind="bar", ax=ax, x="xlabel", xlabel="")

plt.tight_layout()
ax.legend([r"$\mathrm{FCC-ee}$", r"$\mathrm{HL- LHC}$"], frameon=False)
ax.set_ylabel(r"$|\lambda_\phi|$")
ax.set_title("")
# plt.legend(loc='upper right')
plt.savefig("./results/barplot_quadruplet.pdf")

# def plot_uv_posterior_bar(loop_order):
#
#     fig, axes = plt.subplots(figsize=(18, 8), ncols=1, nrows=1)
#
#
#     x_labels = []
#
#     for index, value in bounds_uv.iteritems()
#
#
#
#     total_width = 8
#     bar_width = total_width / 2
#     x = np.linspace(0, 200, len(lhc_bounds))
#     # dx = x[1] - x[0]
#
#     logo = plt.imread("/data/theorie/jthoeve/smefit_release/src/smefit/analyze/logo.png")
#
#     axes.imshow(
#         logo,
#         aspect="auto",
#         transform=axes.transAxes,
#         extent=[0.88, 0.98, 0.75, 0.85],
#         zorder=10,
#     )
#
#
#
#     # Plot the bars
#
#     axes.bar(x - bar_width, lhc_bounds[:, 0], bar_width, align='center', label=r'$\rm{LEP+LHC_{Run-2}}, \:g_{{\rm UV}}=1$', color='C0')
#
#     axes.bar(x - bar_width, lhc_bounds[:, 1], bar_width, align='center', edgecolor='C0',
#                 linestyle='--', alpha=0.3, label=r'$\rm{LEP+LHC_{Run-2}},\:g_{{\rm UV}}=4\pi$')
#
#     axes.bar(x, hllhc_bounds[:, 0], bar_width, align='center', label=r'$\rm{+\:HL-LHC}, \:g_{{\rm UV}}=1$',
#                 color='C1')
#
#     axes.bar(x, hllhc_bounds[:, 1], bar_width, align='center', edgecolor='C1',
#                 linestyle='--', alpha=0.3, label=r'$\rm{+\:HL-LHC}, \:g_{{\rm UV}}=4\pi$')
#
#     axes.bar(x + bar_width, fcc_bounds[:, 0], bar_width, align='center', label=r'$\rm{+\:FCC-ee}, \:g_{{\rm UV}}=1$',
#                 color='C2')
#
#     axes.bar(x + bar_width, fcc_bounds[:, 1], bar_width, align='center', edgecolor='C2',
#                 linestyle='--', alpha=0.3, label=r'$\rm{+\:FCC-ee}, \:g_{{\rm UV}}=4\pi$')
#
#     axes.set_xticks(x, x_labels)
#     axes.set_ylabel(r'$M_{\mathrm{UV}}\:{\rm [TeV]}$')
#     axes.tick_params(axis='x', which='major', pad=15)
#     axes.set_yscale('log')
#
#     axes.legend(
#         frameon=False,
#         ncol=3,
#         loc='upper right',
#         fontsize=18
#     )
#
#     fig.tight_layout()
#
#     fig.savefig(result_dir / "fcc_uv_bounds_esppu.pdf")
# #
#
# def plot_uv_posterior(n_params, collection, mod_nrs, EFT=None, name=None, pQCD=None):
#     plot_nr = 1
#
#     # Calculate number of rows and columns
#     n_cols = max(int(np.ceil(np.sqrt(n_params))), 4)
#     if n_params < n_cols:
#         n_cols = n_params
#
#     n_rows = int(np.ceil(n_params / n_cols))
#
#     fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))
#
#     for mod_nr in mod_nrs:
#         if mod_nr == 5:
#             continue
#         print(mod_nr)
#
#
#
#         # path to posterior with model number
#         if pQCD is None:
#             posterior_path_mod_1 = Path(posterior_path.format(collection, mod_nr, "LO", EFT))
#             posterior_path_mod_2 = Path(posterior_path.format(collection, mod_nr, "NLO", EFT))
#         elif EFT is None:
#
#             posterior_path_mod_1 = Path(posterior_path.format(collection, "lhc", mod_nr, pQCD, "NHO"))
#             posterior_path_mod_2 = Path(posterior_path.format(collection, "hllhc", mod_nr, pQCD, "NHO"))
#             posterior_path_mod_3 = Path(posterior_path.format(collection, "fcc", mod_nr, pQCD, "NHO"))
#         else:
#             print("EFT and pQCD cannot both be None, aborting")
#             sys.exit()
#
#         if posterior_path_mod_1.exists():
#             # Open the JSON file and load its contents
#             try:
#                 with open(posterior_path_mod_1) as f:
#                     posterior_1 = json.load(f)
#
#                 with open(posterior_path_mod_2) as f:
#                     posterior_2 = json.load(f)
#
#                 with open(posterior_path_mod_3) as f:
#                     posterior_3 = json.load(f)
#             except FileNotFoundError:
#                 continue
#
#             for (key, samples_1_list), (_, samples_2_list), (_, samples_3_list) in zip(
#                     posterior_1.items(), posterior_2.items(), posterior_3.items()
#             ):
#                 if not key.startswith("inv"):
#                     continue
#                 else:
#                     samples_1 = np.array(samples_1_list)
#                     samples_2 = np.array(samples_2_list)
#                     samples_3 = np.array(samples_3_list)
#                     ax = plt.subplot(n_rows, n_cols, plot_nr)
#                     plot_nr += 1
#
#                     x_low, x_high = find_xrange([samples_1, samples_2], 0.05)
#
#                     ax.hist(
#                         samples_1[(samples_1 > x_low) & (samples_1 < x_high)],
#                         bins="fd",
#                         density=True,
#                         edgecolor="black",
#                         alpha=0.4,
#                     )
#
#                     ax.hist(
#                         samples_2[(samples_2 > x_low) & (samples_2 < x_high)],
#                         bins="fd",
#                         density=True,
#                         edgecolor="black",
#                         alpha=0.4,
#                     )
#
#                     ax.hist(
#                         samples_3[(samples_3 > x_low) & (samples_3 < x_high)],
#                         bins="fd",
#                         density=True,
#                         edgecolor="black",
#                         alpha=0.4,
#                     )
#
#                     ax.set_xlim(x_low, x_high)
#                     ax.set_ylim(0, ax.get_ylim()[1] + ax.get_ylim()[1] * 0.2)
#
#                     ax.tick_params(which="both", direction="in", labelsize=22.5)
#                     ax.tick_params(labelleft=False)
#
#                     ax.text(
#                         0.05,
#                         0.95,
#                         inv_param_dict[mod_nr][key],
#                         fontsize=17,
#                         ha="left",
#                         va="top",
#                         transform=ax.transAxes
#                     )
#
#                     ax.text(
#                         0.95,
#                         0.95,
#                         mod_dict[mod_nr],
#                         fontsize=20,
#                         ha="right",
#                         va="top",
#                         transform=ax.transAxes
#                     )
#
#
#     if pQCD is None:
#         order_EFT = -2 if EFT == 'NHO' else -4
#         fig.legend([f"$\mathrm{{LO}}\;\mathcal{{O}}\\left(\Lambda^{{{order_EFT}}}\\right)$",
#                     f"$\mathrm{{NLO}}\;\mathcal{{O}}\\left(\Lambda^{{{order_EFT}}}\\right)$"], loc="upper center",
#                    ncol=2,
#                    prop={"size": 25 * (n_cols * 4) / 20}, bbox_to_anchor=(0.5, 1.0), frameon=False)
#
#         if n_rows > 1:
#             plt.tight_layout(rect=[0, 0.05 * (5. / n_rows), 1, 1 - 0.05 * (5. / n_rows)])  # make room for the legend
#         else:
#             plt.tight_layout(rect=[0, 0.08, 1, 0.92])  # make room for the legend
#         if name is not None:
#             fig.savefig(result_dir / "{}_posteriors_LO_vs_NLO_{}_{}.pdf".format(collection, EFT, name))
#         else:
#             fig.savefig(result_dir / "{}_posteriors_LO_vs_NLO_{}.pdf".format(collection, EFT))
#     elif EFT is None:
#         fig.legend([f"$\mathrm{{LHC}}\;\mathcal{{O}}\\left(\Lambda^{{-2}}\\right)$",
#                     f"$\mathrm{{HL-LHC}}\;\mathcal{{O}}\\left(\Lambda^{{-2}}\\right)$",
#                     f"$\mathrm{{FCCee}}\;\mathcal{{O}}\\left(\Lambda^{{-2}}\\right)$"], ncol=3,
#                    prop={"size": 25 * (n_cols * 4) / 20}, bbox_to_anchor=(0.5, 1.0), loc='upper center', frameon=False)
#         if n_rows > 1:
#             plt.tight_layout(rect=[0, 0.05 * (5. / n_rows), 1, 1 - 0.05 * (5. / n_rows)])  # make room for the legend
#         else:
#             plt.tight_layout(rect=[0, 0.08, 1, 0.92])  # make room for the legend
#
#         fig.savefig(result_dir / "{}_posteriors_lhc_vs_hlhc_{}.png".format(collection, name))
#
# plot_uv_posterior_bar(10, pQCD="NLO")
