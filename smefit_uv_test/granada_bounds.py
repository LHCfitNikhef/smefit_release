# -*- coding: utf-8 -*-
import importlib
import json
import pathlib
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from latex_dicts import mod_dict
from latex_dicts import uv_param_dict
from latex_dicts import inv_param_dict
import arviz as az

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc, use

collection = "1L"

here = pathlib.Path(__file__).parent
base_path = pathlib.Path(f"{here.parent}/runcards/uv_models/UV_scan/{collection}/")
sys.path = [str(base_path)] + sys.path

# result dir
result_dir = here / "results"
Path.mkdir(result_dir, parents=True, exist_ok=True)

mod_list = []
for p in base_path.iterdir():
    if p.name.startswith("InvarsFit") and p.suffix == ".py":
        mod_list.append(importlib.import_module(f"{p.stem}"))

use("PDF")
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", **{"usetex": True, "latex.preamble": r"\usepackage{amssymb}"})

# compute the invariants
pQCD = ['LO', 'NLO']
EFT = ['NHO', 'HO']

for model in mod_list:
    for pQCD in ['LO', 'NLO']:
        for EFT in ['NHO', 'HO']:
            print(model)
            model.MODEL_SPECS['pto'] = pQCD
            model.MODEL_SPECS['eft'] = EFT
            invariants = []
            for k, attr in model.__dict__.items():
                if k.startswith('inv'):
                    invariants.append(attr)
            model.inspect_model(model.MODEL_SPECS, invariants)

# Specify the path to the JSON file
posterior_path = f"{here.parent}/results/{{}}_UV_{{}}_{{}}_{{}}_NS/inv_posterior.json"


def nested_dict(num_levels):
    if num_levels == 1:
        return defaultdict(dict)
    else:
        return defaultdict(lambda: nested_dict(num_levels - 1))


def compute_bounds(collection, mod_nrs):
    table_dict = nested_dict(4)
    n_params = 0
    for mod_nr in mod_nrs:
        for pQCD in ["LO", "NLO"]:
            for EFT in ["NHO", "HO"]:
                # path to posterior with model number
                posterior_path_mod = Path(posterior_path.format(collection, mod_nr, pQCD, EFT))
                if posterior_path_mod.exists():

                    # Open the JSON file and load its contents
                    with open(posterior_path_mod) as f:
                        posterior = json.load(f)

                    for key, samples in posterior.items():
                        if not key.startswith("inv"):
                            continue
                        else:
                            table_dict[mod_nr][key][pQCD][EFT] = az.hdi(np.array(samples), hdi_prob=.95)
                            n_params += 1

    return table_dict, int(n_params / 4)


def dict_to_latex_table(nested_dict, mod_dict, uv_param_dict, label, caption):

    table = "\\begin{table}\n"
    table += "\\begin{center}\n"
    table += "\\renewcommand{\\arraystretch}{1.4}"
    table += "\\begin{tabular}{|c|c|c|c|c|c|}\n"
    table += "\\hline\n"
    table += "Model & UV invariants & LO $\\mathcal{O}\\left(\\Lambda^{-2}\\right)$ &LO $\\mathcal{O}\\left(\\Lambda^{-4}\\right)$ & NLO $\\mathcal{O}\\left(\\Lambda^{-2}\\right)$ & NLO $\\mathcal{O}\\left(\\Lambda^{-4}\\right)$ \\\\\n"
    table += "\\hline\n"

    for model_nr, params in nested_dict.items():
        model_row = True
        for parameter, param_dict in params.items():
            LO_NHO = [np.round(x, 3) for x in param_dict["LO"]["NHO"]]
            LO_HO = [np.round(x, 3) for x in param_dict["LO"]["HO"]]
            NLO_NHO = [np.round(x, 3) for x in param_dict["NLO"]["NHO"]]
            NLO_HO = [np.round(x, 3) for x in param_dict["NLO"]["HO"]]

            if model_row:

                table += f"{mod_dict[model_nr]} & {inv_param_dict[model_nr][parameter]} & {LO_NHO} & {LO_HO} & {NLO_NHO} & {NLO_HO} \\\\\n"
                model_row = False
            else:
                table += f" & {inv_param_dict[model_nr][parameter]} & {LO_NHO} & {LO_HO} & {NLO_NHO} & {NLO_HO}  \\\\\n"

    table += "\\hline\n"
    table += "\\end{tabular}\n"
    table += f"\\caption{{{caption}}}\n"
    table += "\\end{center}\n"
    table += "\\end{table}\n"


    return table


scalar_mdl_nrs = range(21)
vboson_mdl_nrs = range(21, 37)
vfermion_mdl_nrs = range(37, 50)
mp_mdl_idx = ["Q1_Q7_W1"]
oneloop_mdl_idx = [5]

# scalar_dict, n_scalars = compute_bounds("Granada", scalar_mdl_nrs)
# vector_boson_dict, n_vbosons = compute_bounds("Granada", vboson_mdl_nrs)
# vector_fermion_dict, n_vfermions = compute_bounds("Granada", vfermion_mdl_nrs)
# mp_dict, n_mp = compute_bounds("MP", mp_mdl_idx)
oneloop_dict, n_scalars_1L = compute_bounds(collection, oneloop_mdl_idx)

# latex_table_scalar = dict_to_latex_table(
#     scalar_dict,
#     mod_dict,
#     uv_param_dict,
#     "cl-heavy-scalar",
#     "95\\% CL intervals of the heavy scalar fields UV couplings.",
# )
# latex_table_vboson = dict_to_latex_table(
#     vector_boson_dict,
#     mod_dict,
#     uv_param_dict,
#     "cl-heavy-vboson",
#     "95\\% CL intervals of the heavy vector boson fields UV couplings.",
# )
# latex_table_vfermion = dict_to_latex_table(
#     vector_fermion_dict,
#     mod_dict,
#     uv_param_dict,
#     "cl-heavy-vfermion",
#     "95\\% CL intervals of the heavy vector fermion fields UV couplings.",
# )
# latex_table_mp = dict_to_latex_table(
#     mp_dict,
#     mod_dict,
#     uv_param_dict,
#     "cl-mp-model",
#     "95\\% CL intervals of the UV couplings that enter in the multiparticle model.",
# )
latex_table_1l = dict_to_latex_table(
    oneloop_dict,
    mod_dict,
    uv_param_dict,
    "cl-1l-model",
    "95\\% CL intervals of the UV couplings in \\varphi at one-loop.",
)

#Save LaTeX code to a .tex file
# with open(result_dir / "table_scalar.tex", "w") as file:
#     file.write(latex_table_scalar)
# with open(result_dir / "table_vboson.tex", "w") as file:
#     file.write(latex_table_vboson)
# with open(result_dir / "table_vfermion.tex", "w") as file:
#     file.write(latex_table_vfermion)
# with open(result_dir / "table_mp.tex", "w") as file:
#     file.write(latex_table_mp)
with open(result_dir / "table_1l.tex", "w") as file:
    file.write(latex_table_1l)


def find_xrange(samples):

    x_low = []
    x_high = []

    for sample in samples:

        n, bins = np.histogram(
            sample,
            bins="fd",
            density=True,

        )

        max_index = np.argmax(n)
        threshold = 0.05 * n[max_index]
        indices = np.where(n >= threshold)[0]

        low, up = az.hdi(sample, hdi_prob=.95)

        index_max = np.max(indices)
        index_min = np.min(indices)

        x_high.append(bins[index_max])

        if low < 0:
            x_low.append(bins[index_min])
        else:
            x_low.append(- 0.1 * bins[index_max])

    return min(x_low), max(x_high)


    # min_low = min(low_1, low_2)
    #
    #
    #
    #
    # # if n_1[max_index_1] > n_2[max_index_2]:
    # #     x_low = bins_1[index_min_1]
    # #     x_high = bins_1[index_max_1]
    # # else:
    # #     x_low = bins_2[index_min_2]
    # #     x_high = bins_2[index_max_2]
    # #
    # # if min_low > 0:
    # #     x_low = -0.1 * x_high
    #
    #
    # x_high_2 = bins_2[index_max_2]
    #
    # if min_low < 0:
    #     x_low_1 = bins_1[index_min_1]
    #     x_low_2 = bins_2[index_min_2]
    # else:
    #     x_low_1 = - 0.1 * x_high_1
    #     x_low_2 = - 0.1 * x_high_2
    #
    # x_low = min(x_low_1, x_low_2)
    # x_high = max(x_high_1, x_high_2)

def plot_uv_posterior(n_params, collection, mod_nrs, EFT=None, name=None, pQCD=None):
    plot_nr = 1

    # Calculate number of rows and columns
    n_cols = max(int(np.ceil(np.sqrt(n_params))), 4)
    if n_params < n_cols:
        n_cols = n_params
    n_rows = int(np.ceil(n_params / n_cols))

    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))

    for mod_nr in mod_nrs:
        print(mod_nr)

        # path to posterior with model number
        if pQCD is None:
            posterior_path_mod_1 = Path(posterior_path.format(collection, mod_nr, "LO", EFT))
            posterior_path_mod_2 = Path(posterior_path.format(collection, mod_nr, "NLO", EFT))
        elif EFT is None:
            posterior_path_mod_1 = Path(posterior_path.format(collection, mod_nr, pQCD, "NHO"))
            posterior_path_mod_2 = Path(posterior_path.format(collection, mod_nr, pQCD, "HO"))
        else:
            print("EFT and pQCD cannot both be None, aborting")
            sys.exit()

        if posterior_path_mod_1.exists():
            # Open the JSON file and load its contents
            with open(posterior_path_mod_1) as f:
                posterior_1 = json.load(f)

            with open(posterior_path_mod_2) as f:
                posterior_2 = json.load(f)

            for (key, samples_1_list), (_, samples_2_list) in zip(
                    posterior_1.items(), posterior_2.items()
            ):
                if not key.startswith("inv"):
                    continue
                else:
                    samples_1 = np.array(samples_1_list)
                    samples_2 = np.array(samples_2_list)
                    ax = plt.subplot(n_rows, n_cols, plot_nr)
                    plot_nr += 1

                    x_low, x_high = find_xrange([samples_1, samples_2])

                    ax.hist(
                        samples_1[(samples_1 > x_low) & (samples_1 < x_high)],
                        bins="fd",
                        density=True,
                        edgecolor="black",
                        alpha=0.4,
                    )

                    ax.hist(
                        samples_2[(samples_2 > x_low) & (samples_2 < x_high)],
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
                        inv_param_dict[mod_nr][key],
                        fontsize=20,
                        ha="left",
                        va="top",
                    )

    if pQCD is None:
        order_EFT = -2 if EFT == 'NHO' else -4
        fig.legend([f"$\mathrm{{LO}}\;\mathcal{{O}}\\left(\Lambda^{{{order_EFT}}}\\right)$",
                    f"$\mathrm{{NLO}}\;\mathcal{{O}}\\left(\Lambda^{{{order_EFT}}}\\right)$"], loc="upper center", ncol=2,
                   prop={"size": 25 * (n_cols * 4) / 20}, bbox_to_anchor=(0.5, 1.0), frameon=False)

        #plt.tight_layout(rect=[0, 0.05 * (5./n_rows), 1, 1 - 0.05 * (5./n_rows)])  # make room for the legend
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # make room for the legend
        if name is not None:
            fig.savefig(result_dir / "{}_posteriors_LO_vs_NLO_{}_{}.pdf".format(collection, EFT, name))
        else:
            fig.savefig(result_dir / "{}_posteriors_LO_vs_NLO_{}.pdf".format(collection, EFT))
    elif EFT is None:
        fig.legend([f"$\mathrm{{{pQCD}}}\;\mathcal{{O}}\\left(\Lambda^{{-2}}\\right)$",
                    f"$\mathrm{{{pQCD}}}\;\mathcal{{O}}\\left(\Lambda^{{-4}}\\right)$"], ncol=2,
                   prop={"size": 25 * (n_cols * 4) / 20}, bbox_to_anchor=(0.5, 1.0), loc='upper center', frameon=False)

        #plt.tight_layout(rect=[0, 0.05 * (5./n_rows), 1, 1 - 0.05 * (5./n_rows)]) # make room for the legend
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # make room for the legend

        if name is not None:
            fig.savefig(result_dir / "{}_posteriors_HO_vs_NHO_{}_{}.pdf".format(collection, pQCD, name))
        else:
            fig.savefig(result_dir / "{}_posteriors_HO_vs_NHO_{}.pdf".format(collection, pQCD))


# plot_uv_posterior(n_mp, "MP", mp_mdl_idx, EFT="NHO")
# plot_uv_posterior(n_mp, "MP", mp_mdl_idx, EFT="HO")
# plot_uv_posterior(n_mp, "MP",mp_mdl_idx, pQCD="LO")
# plot_uv_posterior(n_mp, "MP", mp_mdl_idx, pQCD="NLO")
#
# plot_uv_posterior(n_vbosons, "Granada", vboson_mdl_nrs, EFT="NHO", name="vbosons")
# plot_uv_posterior(n_vbosons, "Granada", vboson_mdl_nrs, EFT="HO", name="vbosons")
# plot_uv_posterior(n_vfermions, "Granada", vfermion_mdl_nrs, EFT="NHO", name="vfermions")
# plot_uv_posterior(n_vfermions, "Granada", vfermion_mdl_nrs, EFT="HO", name="vfermions")
# plot_uv_posterior(n_scalars, "Granada", scalar_mdl_nrs, EFT="NHO", name="scalars")
# plot_uv_posterior(n_scalars, "Granada", scalar_mdl_nrs, EFT="HO", name="scalars")
#
# plot_uv_posterior(n_vbosons, "Granada", vboson_mdl_nrs, name="vbosons", pQCD="LO")
# plot_uv_posterior(n_vbosons, "Granada", vboson_mdl_nrs, name="vbosons", pQCD="NLO")
# plot_uv_posterior(n_vfermions, "Granada", vfermion_mdl_nrs, name="vfermions", pQCD="LO")
# plot_uv_posterior(n_vfermions, "Granada", vfermion_mdl_nrs, name="vfermions", pQCD="NLO")
# plot_uv_posterior(n_scalars, "Granada", scalar_mdl_nrs,name="scalars", pQCD="LO")
# plot_uv_posterior(n_scalars, "Granada", scalar_mdl_nrs, name="scalars", pQCD="NLO")

plot_uv_posterior(n_scalars_1L, "1L", oneloop_mdl_idx, EFT="NHO", name="scalars")
plot_uv_posterior(n_scalars_1L, "1L", oneloop_mdl_idx, EFT="HO", name="scalars")
plot_uv_posterior(n_scalars_1L, "1L", oneloop_mdl_idx, pQCD="LO", name="scalars")
plot_uv_posterior(n_scalars_1L, "1L", oneloop_mdl_idx, pQCD="NLO", name="scalars")


for file in result_dir.iterdir():
    if file.suffix == ".pdf":
        subprocess.run(["pdfcrop", str(file), str(file)])
