# -*- coding: utf-8 -*-
import importlib
import json
import pathlib
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc, use

here = pathlib.Path(__file__).parent
base_path = pathlib.Path(f"{here.parent}/runcards/uv_models/UV_scan/Granada/")
sys.path = [str(base_path)] + sys.path


mod_list = []
for p in base_path.iterdir():
    if p.suffix == "py":
        mod_list.append(importlib.import_module(f"{p.stem}"))

use("PDF")
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", **{"usetex": True, "latex.preamble": r"\usepackage{amssymb}"})

# Specify the path to the JSON file
posterior_path = f"{here.parent}/results/Granada_UV_{{}}_{{}}_{{}}_NS/posterior.json"

mod_dict = {
    2: r"$\mathcal{S}$",
    3: r"$\mathcal{S}_1$",
    5: r"$\varphi$",
    6: r"$\Xi$",
    7: r"$\Xi_1$",
    10: r"$\omega_1$",
    12: r"$\omega_4$",
    15: r"$\zeta$",
    16: r"$\Omega_1$",
    18: r"$\Omega_4$",
    19: r"$\Upsilon$",
    20: r"$\Phi$",
    21: r"$\mathcal{B}$",
    22: r"$\mathcal{B}_1$",
    23: r"$\mathcal{W}_1$",
    24: r"$\mathcal{W}_1$",
    25: r"$\mathcal{G}$",
    27: r"$\mathcal{H}$",
    33: r"$\mathcal{Q}_5$",
    36: r"$\mathcal{Y}_5$",
    37: r"$N$",
    38: r"$E$",
    39: r"$\Delta_1$",
    40: r"$\Delta_3$",
    41: r"$\Sigma$",
    42: r"$\Sigma_1$",
    43: r"$U$",
    44: r"$D$",
    45: r"$Q_1$",
    47: r"$Q_y$",
    48: r"$T_1$",
    49: r"$T_2$",
}

uv_param_dict = {
    "kS": r"$\kappa_{\mathcal{S}}$",
    "yS1f12": r"$\left(y_{\mathcal{S}_1}\right)_{12}$",
    "yS1f21": r"$\left(y_{\mathcal{S}_1}\right)_{21}$",
    "lamVarphi": r"$\lambda_{\varphi}$",
    "yVarphiuf33": r"$\left(y_{\varphi}^u\right)_{33}$",
    "kXi": r"$\kappa_{\Xi}$",
    "kXi1": r"$\kappa_{\Xi1}$",
    "yomega1qqf33": r"$\left(y_{\omega_1}^{qq}\right)_{33}$",
    "yomega4uuf33": r"$\left(y_{\omega_4}^{uu}\right)_{33}$",
    "yZetaqqf33": r"$\left(y_{\zeta}^{qq}\right)_{33}$",
    "yOmega1qqf33": r"$\left(y_{\Omega_1}^{qq}\right)_{33}$",
    "yOmega4f33": r"$\left(y_{\Omega_4}\right)_{33}$",
    "yUpsf33": r"$\left(y_{\Upsilon}\right)_{33}$",
    "yPhiquf33": r"$\left(y_{\Phi}^{qu}\right)_{33}$",
    "gBH": r"$g_B^\phi$",
    "gBdf11": r"$\left(g_B^d\right)_{11}$",
    "gBef11": r"$\left(g_B^e\right)_{11}$",
    "gBef22": r"$\left(g_B^e\right)_{22}$",
    "gBef33": r"$\left(g_B^e\right)_{33}$",
    "gBqf11": r"$\left(g_B^q\right)_{11}$",
    "gBqf33": r"$\left(g_B^q\right)_{33}$",
    "gBuf11": r"$\left(g_B^u\right)_{11}$",
    "gBuf33": r"$\left(g_B^u\right)_{33}$",
    "gB1H": r"$g_{B_1}^\phi$",
    "gWH": r"$g_{\mathcal{W}}^\phi$",
    "gWLf11": r"$\left(g_{\mathcal{W}}^l\right)_{11}$",
    "gWLf22": r"$\left(g_{\mathcal{W}}^l\right)_{22}$",
    "gWLf33": r"$\left(g_{\mathcal{W}}^l\right)_{33}$",
    "gWqf11": r"$\left(g_{\mathcal{W}}^q\right)_{11}$",
    "gWqf33": r"$\left(g_{\mathcal{W}}^q\right)_{33}$",
    "gW1H": r"$g_{\mathcal{W}_1}^\phi$",
    "gGdf11": r"$\left(g_{\mathcal{G}}^d\right)_{11}$",
    "gGqf33": r"$\left(g_{\mathcal{G}}^q\right)_{33}$",
    "gGuf33": r"$\left(g_{\mathcal{G}}^u\right)_{33}$",
    "gHf33": r"$\left(g_{\mathcal{H}}\right)_{33}$",
    "gQv5uqf33": r"$\left(g_{\mathcal{Q}_5}^{uq}\right)_{33}$",
    "gY5f33": r"$\left(g_{\mathcal{Y}_5}\right)_{33}$",
    "lamNef3": r"$\left(\lambda_N^e\right)_3$",
    "lamEff3": r"$\left(\lambda_E\right)_3$",
    "lamDelta1f3": r"$\left(\lambda_{\Delta_1}\right)_3$",
    "lamDelta3f3": r"$\left(\lambda_{\Delta_3}\right)_3$",
    "lamSigmaf3": r"$\left(\lambda_{\Sigma}\right)_3$",
    "lamSigma1f3": r"$\left(\lambda_{\Sigma_1}\right)_3$",
    "lamUf3": r"$\left(\lambda_{U}\right)_3$",
    "lamDff3": r"$\left(\lambda_{D}\right)_3$",
    "lamQ1uf3": r"$\left(\lambda_{\mathcal{Q}_1}^u\right)_3$",
    "lamQ7f3": r"$\left(\lambda_{\mathcal{Q}_7}\right)_3$",
    "lamT1f3": r"$\left(\lambda_{T_1}\right)_3$",
    "lamT2f3": r"$\left(\lambda_{T_2}\right)_3$",
}


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
                        if key.startswith("O"):
                            continue
                        else:
                            table_dict[mod_nr][key][pQCD][EFT] = [
                                np.percentile(samples, 2.5),
                                np.percentile(samples, 97.5),
                            ]
                            n_params += 1

    return table_dict, int(n_params / 4)


def dict_to_latex_table(nested_dict, mod_dict, uv_param_dict, label, caption):

    table = "\\documentclass{article}\\n"
    table += "\\usepackage[a4paper]{geometry}\\n"
    table += "\\begin{document}\n"
    table += "\\begin{table}\n"
    table += "\\begin{center}\n"
    table += "\\renewcommand{\\arraystretch}{1.4}"
    table += "\\begin{tabular}{|c|c|c|c|c|c|}\n"
    table += "\\hline\n"
    table += "Model & UV coupling & LO \\mathcal{O}\\left(\\Lambda^{-2}\\right) &LO \\mathcal{O}\\left(\\Lambda^{-4}\\right) & NLO \\mathcal{O}\\left(\\Lambda^{-2}\\right) & NLO \\mathcal{O}\\left(\\Lambda^{-4}\\right) \\\\\n"
    table += "\\hline\n"

    for model_nr, params in nested_dict.items():
        model_row = True
        for parameter, param_dict in params.items():
            LO_NHO = [round(x, 2) for x in param_dict["LO"]["NHO"]]
            LO_HO = [round(x, 2) for x in param_dict["LO"]["HO"]]
            NLO_NHO = [round(x, 2) for x in param_dict["NLO"]["NHO"]]
            NLO_HO = [round(x, 2) for x in param_dict["NLO"]["HO"]]

            if model_row:

                table += f"{mod_dict[model_nr]} & {uv_param_dict[parameter]} & {LO_NHO} & {LO_HO} & {NLO_NHO} & {NLO_HO} \\\\\n"
                model_row = False
            else:
                table += f" & {uv_param_dict[parameter]} & {LO_NHO} & {LO_HO} & {NLO_NHO} & {NLO_HO}  \\\\\n"

    table += "\\hline\n"
    table += "\\end{tabular}\n"
    table += f"\\caption{{{caption}}}\\n"
    table += "\\end{center}\n"
    table += "\\end{table}\n"
    table += "\\end{document}\n"

    return table


scalar_mdl_nrs = range(21)
vboson_mdl_nrs = range(21, 37)
vfermion_mdl_nrs = range(37, 50)

scalar_dict, n_scalars = compute_bounds(scalar_mdl_nrs)
vector_boson_dict, n_vbosons = compute_bounds(vboson_mdl_nrs)
vector_fermion_dict, n_vfermions = compute_bounds(vfermion_mdl_nrs)

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

# Save LaTeX code to a .tex file
with open("table_scalar.tex", "w") as file:
    file.write(latex_table_scalar)
with open("table_vboson.tex", "w") as file:
    file.write(latex_table_vboson)
with open("table_vfermion.tex", "w") as file:
    file.write(latex_table_vfermion)

# Compile LaTeX code to PDF
subprocess.run(["pdflatex", "-interaction=batchmode", "table_scalar.tex"])
subprocess.run(["rm", "table_scalar.aux", "table_scalar.log"])
subprocess.run(["pdflatex", "-interaction=batchmode", "table_vboson.tex"])
subprocess.run(["rm", "table_vboson.aux", "table_vboson.log"])
subprocess.run(["pdflatex", "-interaction=batchmode", "table_vfermion.tex"])
subprocess.run(["rm", "table_vfermion.aux", "table_vfermion.log"])
# Rename the output file


def plot_uv_posterior(n_params, mod_nrs, EFT, name):

    plot_nr = 1

    # Calculate number of rows and columns
    n_rows = int(np.ceil(np.sqrt(n_params)))
    n_cols = int(np.ceil(n_params / n_rows))

    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))

    for mod_nr in mod_nrs:
        print(mod_nr)
        if mod_nr == 3:
            continue
        # path to posterior with model number
        posterior_path_mod_lo = Path(posterior_path.format(mod_nr, "LO", EFT))
        posterior_path_mod_nlo = Path(posterior_path.format(mod_nr, "NLO", EFT))
        if posterior_path_mod_lo.exists():
            # Open the JSON file and load its contents
            with open(posterior_path_mod_lo) as f:
                posterior_lo = json.load(f)

            with open(posterior_path_mod_nlo) as f:
                posterior_nlo = json.load(f)

            for (key, samples_lo), (_, samples_nlo) in zip(
                posterior_lo.items(), posterior_nlo.items()
            ):
                print(key)
                if key.startswith("O"):
                    continue
                else:
                    ax = plt.subplot(n_rows, n_cols, plot_nr)
                    plot_nr += 1

                    ax.hist(
                        samples_lo,
                        bins="fd",
                        density=True,
                        edgecolor="black",
                        alpha=0.4,
                    )

                    ax.hist(
                        samples_nlo,
                        bins="fd",
                        density=True,
                        edgecolor="black",
                        alpha=0.4,
                    )

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
                        f"{uv_param_dict[key]}",
                        fontsize=20,
                        ha="left",
                        va="top",
                    )

    fig.legend(["LO " + EFT, "NLO " + EFT], loc="lower center", fontsize=25, ncol=2)

    fig.savefig("./granada_posteriors_{}_{}.pdf".format(EFT, name))


plot_uv_posterior(n_vbosons, vboson_mdl_nrs, "NHO", "vbosons")
plot_uv_posterior(n_vbosons, vboson_mdl_nrs, "HO", "vbosons")
plot_uv_posterior(n_vfermions, vfermion_mdl_nrs, "NHO", "vfermions")
plot_uv_posterior(n_vfermions, vfermion_mdl_nrs, "HO", "vfermions")
plot_uv_posterior(n_scalars, scalar_mdl_nrs, "NHO", "scalars")
plot_uv_posterior(n_scalars, scalar_mdl_nrs, "HO", "scalars")

subprocess.run(
    [
        "gs",
        "-dBATCH",
        "-dNOPAUSE",
        "-q",
        "-sDEVICE=pdfwrite",
        "-sOutputFile=models_granada.pdf",
        "table_scalar.pdf",
        "granada_posteriors_NHO_scalars.pdf",
        "granada_posteriors_HO_scalars.pdf",
        "table_vboson.pdf",
        "granada_posteriors_NHO_vbosons.pdf",
        "granada_posteriors_HO_vbosons.pdf",
        "table_vfermion.pdf",
        "granada_posteriors_NHO_vfermions.pdf",
        "granada_posteriors_HO_vfermions.pdf",
    ]
)
