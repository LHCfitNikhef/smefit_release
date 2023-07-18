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

here = pathlib.Path(__file__).parent
base_path = pathlib.Path(f"{here.parent}/runcards/uv_models/UV_scan/MP/")
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
            model.inspect_model(model.MODEL_SPECS, model.build_uv_posterior, invariants, model.check_constrain)

# Specify the path to the JSON file
posterior_path = f"{here.parent}/results/Granada_UV_{{}}_{{}}_{{}}_NS/inv_posterior.json"


def nested_dict(num_levels):
    if num_levels == 1:
        return defaultdict(dict)
    else:
        return defaultdict(lambda: nested_dict(num_levels - 1))


def compute_bounds(mod_nrs):
    table_dict = nested_dict(4)
    n_params = 0
    for mod_nr in mod_nrs:
        for pQCD in ["LO", "NLO"]:
            for EFT in ["NHO", "HO"]:
                # path to posterior with model number
                posterior_path_mod = Path(posterior_path.format(mod_nr, pQCD, EFT))

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
    # table = "\\documentclass{article}\\n"
    # table += "\\usepackage[a4paper]{geometry}\\n"
    # table += "\\begin{document}\n"
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
    # table += "\\end{document}\n"

    return table


scalar_mdl_nrs = range(21)
vboson_mdl_nrs = range(21, 37)
vfermion_mdl_nrs = range(37, 50)
mp_mdl_idx = ["Q1_Q7_W1"]

scalar_dict, n_scalars = compute_bounds(scalar_mdl_nrs)
vector_boson_dict, n_vbosons = compute_bounds(vboson_mdl_nrs)
vector_fermion_dict, n_vfermions = compute_bounds(vfermion_mdl_nrs)
mp_dict, n_mp = compute_bounds(mp_mdl_idx)

latex_table_scalar = dict_to_latex_table(
    scalar_dict,
    mod_dict,
    uv_param_dict,
    "cl-heavy-scalar",
    "95\\% CL intervals of the heavy scalar fields UV couplings.",
)
latex_table_vboson = dict_to_latex_table(
    vector_boson_dict,
    mod_dict,
    uv_param_dict,
    "cl-heavy-vboson",
    "95\\% CL intervals of the heavy vector boson fields UV couplings.",
)
latex_table_vfermion = dict_to_latex_table(
    vector_fermion_dict,
    mod_dict,
    uv_param_dict,
    "cl-heavy-vfermion",
    "95\\% CL intervals of the heavy vector fermion fields UV couplings.",
)
latex_table_mp = dict_to_latex_table(
    mp_dict,
    mod_dict,
    uv_param_dict,
    "cl-mp-model",
    "95\\% CL intervals of the UV couplings that enter in the multiparticle model.",
)

#Save LaTeX code to a .tex file
with open(result_dir / "table_scalar.tex", "w") as file:
    file.write(latex_table_scalar)
with open(result_dir / "table_vboson.tex", "w") as file:
    file.write(latex_table_vboson)
with open(result_dir / "table_vfermion.tex", "w") as file:
    file.write(latex_table_vfermion)
with open(result_dir / "table_mp.tex", "w") as file:
    file.write(latex_table_mp)



# Compile LaTeX code to PDF

# subprocess.run(["pdflatex", "-interaction=batchmode", "-output-directory",  str(result_dir), str(result_dir/ "table_scalar.tex")])
# subprocess.run(["rm", str(result_dir / "table_scalar.aux"), str(result_dir / "table_scalar.log")])
# subprocess.run(["pdflatex", "-interaction=batchmode", "-output-directory",  str(result_dir), str(result_dir/ "table_vboson.tex")])
# subprocess.run(["rm", str(result_dir / "table_vboson.aux"), str(result_dir / "table_vboson.log")])
# subprocess.run(["pdflatex", "-interaction=batchmode", "-output-directory",  str(result_dir), str(result_dir/ "table_vfermion.tex")])
# subprocess.run(["rm", str(result_dir / "table_vfermion.aux"), str(result_dir / "table_vfermion.log")])


def plot_uv_posterior(n_params, mod_nrs, EFT, name=None, pQCD=None):
    plot_nr = 1

    # Calculate number of rows and columns
    n_cols = max(int(np.ceil(np.sqrt(n_params))), 4)
    n_rows = int(np.ceil(n_params / n_cols))

    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))

    for mod_nr in mod_nrs:
        print(mod_nr)

        # path to posterior with model number
        if pQCD is None:
            posterior_path_mod_1 = Path(posterior_path.format(mod_nr, "LO", EFT))
            posterior_path_mod_2 = Path(posterior_path.format(mod_nr, "NLO", EFT))
        else:
            posterior_path_mod_1 = Path(posterior_path.format(mod_nr, pQCD, "NHO"))
            posterior_path_mod_2 = Path(posterior_path.format(mod_nr, pQCD, "HO"))

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

                    n_1, bins_1 = np.histogram(
                        samples_1,
                        bins="fd",
                        density=True,

                    )

                    n_2, bins_2 = np.histogram(
                        samples_2,
                        bins="fd",
                        density=True,

                    )

                    max_index_1 = np.argmax(n_1)
                    threshold_1 = 0.05 * n_1[max_index_1]
                    max_index_2 = np.argmax(n_2)
                    threshold_2 = 0.05 * n_2[max_index_2]
                    indices_1 = np.where(n_1 >= threshold_1)[0]
                    indices_2 = np.where(n_2 >= threshold_2)[0]

                    low_1, up_1 = az.hdi(samples_1, hdi_prob=.95)
                    low_2, up_2 = az.hdi(samples_2, hdi_prob=.95)

                    min_low = min(low_1, low_2)
                    max_up = max(up_1, up_2)
                    delta = 0.1 * (max_up - min_low)

                    index_max_1 = np.max(indices_1)
                    index_min_1 = np.min(indices_1)
                    index_max_2 = np.max(indices_2)
                    index_min_2 = np.min(indices_2)

                    if min_low < 0:
                        x_low_1 = bins_1[index_min_1]
                    x_high_1 = bins_1[index_max_1]
                    if min_low < 0:
                        x_low_2 = bins_2[index_min_2]
                    x_high_2 = bins_2[index_max_2]

                    if min_low > 0:
                        x_low_1 = - 0.1 * x_high_1
                        x_low_2 = - 0.1 * x_high_2
                        x_low = min(x_low_1, x_low_2)
                    else:
                        x_low = min(x_low_1, x_low_2)
                    x_high = max(x_high_1, x_high_2)


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
                    #ax.set_xlim(min_lim - 0.1 * (max_lim - min_lim), max_lim + 0.1 * (max_lim - min_lim))

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
                    f"$\mathrm{{NLO}}\;\mathcal{{O}}\\left(\Lambda^{{{order_EFT}}}\\right)$"], loc="upper center",
                   prop={"size": 20}, ncol=2)
        plt.subplots_adjust(top=0.90, wspace=0.1, hspace=0.15)

        if name is not None:
            fig.savefig(result_dir / "granada_posteriors_LO_vs_NLO_{}_{}.pdf".format(EFT, name))
        else:
            fig.savefig(result_dir / "granada_posteriors_LO_vs_NLO_{}.pdf".format(EFT))
    else:
        fig.legend([f"$\mathrm{{{pQCD}}}\;\mathcal{{O}}\\left(\Lambda^{{-2}}\\right)$",
                    f"$\mathrm{{{pQCD}}}\;\mathcal{{O}}\\left(\Lambda^{{-4}}\\right)$"], loc="upper center", ncol=2,
                   prop={"size": 20})
        plt.subplots_adjust(top=0.90, wspace=0.1, hspace=0.1)
        if name is not None:
            fig.savefig(result_dir / "granada_posteriors_HO_vs_NHO_{}_{}.pdf".format(pQCD, name))
        else:
            fig.savefig(result_dir / "granada_posteriors_HO_vs_NHO_{}.pdf".format(pQCD))




# plot_uv_posterior(n_mp, mp_mdl_idx, "NHO")
# plot_uv_posterior(n_mp, mp_mdl_idx, "HO")
# plot_uv_posterior(n_mp, mp_mdl_idx, "NHO", pQCD="LO")
# plot_uv_posterior(n_mp, mp_mdl_idx, "HO", pQCD="NLO")

plot_uv_posterior(n_vbosons, vboson_mdl_nrs, "NHO", "vbosons")
plot_uv_posterior(n_vbosons, vboson_mdl_nrs, "HO", "vbosons")
plot_uv_posterior(n_vfermions, vfermion_mdl_nrs, "NHO", "vfermions")
plot_uv_posterior(n_vfermions, vfermion_mdl_nrs, "HO", "vfermions")
plot_uv_posterior(n_scalars, scalar_mdl_nrs, "NHO", "scalars")
plot_uv_posterior(n_scalars, scalar_mdl_nrs, "HO", "scalars")
#
plot_uv_posterior(n_vbosons, vboson_mdl_nrs, "NHO", "vbosons", pQCD="LO")
plot_uv_posterior(n_vbosons, vboson_mdl_nrs, "HO", "vbosons", pQCD="NLO")
plot_uv_posterior(n_vfermions, vfermion_mdl_nrs, "NHO", "vfermions", pQCD="LO")
plot_uv_posterior(n_vfermions, vfermion_mdl_nrs, "HO", "vfermions", pQCD="NLO")
plot_uv_posterior(n_scalars, scalar_mdl_nrs, "NHO", "scalars", pQCD="LO")
plot_uv_posterior(n_scalars, scalar_mdl_nrs, "HO", "scalars", pQCD="NLO")
#
for file in result_dir.iterdir():
    if file.suffix == ".pdf":
        subprocess.run(["pdfcrop", str(file), str(file)])

# uncomment to merge all pdf files
# merge_pdf = [
#         "gs",
#         "-dBATCH",
#         "-dNOPAUSE",
#         "-q",
#         "-sDEVICE=pdfwrite",
#         "-sOutputFile=models_granada.pdf",
#         "table_scalar.pdf",
#         "granada_posteriors_LO_vs_NLO_NHO_scalars.pdf",
#         "granada_posteriors_LO_vs_NLO_HO_scalars.pdf",
#         "granada_posteriors_HO_vs_NHO_LO_scalars.pdf",
#         "granada_posteriors_HO_vs_NHO_NLO_scalars.pdf",
#         "table_vboson.pdf",
#         "granada_posteriors_LO_vs_NLO_NHO_vbosons.pdf",
#         "granada_posteriors_LO_vs_NLO_HO_vbosons.pdf",
#         "granada_posteriors_HO_vs_NHO_LO_vbosons.pdf",
#         "granada_posteriors_HO_vs_NHO_NLO_vbosons.pdf",
#         "table_vfermion.pdf",
#         "granada_posteriors_LO_vs_NLO_NHO_vfermions.pdf",
#         "granada_posteriors_LO_vs_NLO_HO_vfermions.pdf",
#         "granada_posteriors_HO_vs_NHO_LO_vfermions.pdf",
#         "granada_posteriors_HO_vs_NHO_NLO_vfermions.pdf"
#     ]
# subprocess.run(
#     merge_pdf, cwd=str(result_dir)
# )
