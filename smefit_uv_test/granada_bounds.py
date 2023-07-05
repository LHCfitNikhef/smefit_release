import matplotlib.pyplot as plt
import json
from pathlib import Path
import subprocess
import numpy as np
from matplotlib import rc, use
from collections import defaultdict

use("PDF")
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", **{"usetex": True, "latex.preamble": r"\usepackage{amssymb}"})

# Specify the path to the JSON file
posterior_path = "/data/theorie/jthoeve/smefit_release/results/Granada_UV_{}_{}_{}_NS/posterior.json"

mod_dict = {2: r"$\mathcal{S}$", 3: r"$\mathcal{S}_1$", 5: r"$\varphi$", 6: r"$\Xi$", 7: r"$\Xi_1$", 10: r"$\omega_1$",
            12: r"$\omega_4$", 15: r"$\zeta$", 16: r"$\Omega_1$", 18: r"$\Omega_4$", 19: r"$\Upsilon$", 20: r"$\Phi$"}

uv_param_dict = {'kS': r"$\kappa_{\mathcal{S}}$",
                 'yS1f12': r"$\left(y_{\mathcal{S}_1}\right)_{12}$",
                 'yS1f21': r"$\left(y_{\mathcal{S}_1}\right)_{21}$",
                 'lamVarphi': r"$\lambda_{\varphi}$",
                 'yVarphiuf33': r"$\left(y_{\varphi}^u\right)_{33}$",
                 'kXi': r"$\kappa_{\Xi}$",
                 'kXi1': r"$\kappa_{\Xi1}$",
                 'yomega1qqf33': r"$\left(y_{\omega_1}^{qq}\right)_{33}$",
                 'yomega4uuf33': r"$\left(y_{\omega_4}^{uu}\right)_{33}$",
                 'yZetaqqf33': r"$\left(y_{\zeta}^{qq}\right)_{33}$",
                 'yOmega1qqf33': r"$\left(y_{\Omega_1}^{qq}\right)_{33}$",
                 'yOmega4f33': r"$\left(y_{\Omega_4}\right)_{33}$",
                 'yUpsf33': r"$\left(y_{\Upsilon}\right)_{33}$",
                 'yPhiquf33': r"$\left(y_{\Phi}^{qu}\right)_{33}$"
                 }


def nested_dict(num_levels):
    if num_levels == 1:
        return defaultdict(dict)
    else:
        return defaultdict(lambda: nested_dict(num_levels - 1))


table_dict = nested_dict(4)

for mod_nr in range(21):
    for pQCD in ['LO', 'NLO']:
        for EFT in ['NHO', 'HO']:
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
                        table_dict[mod_nr][key][pQCD][EFT] = [np.percentile(samples, 2.5), np.percentile(samples, 97.5)]


def nested_dict_to_latex_table(nested_dict):
    table = "\\documentclass{article}\\n"
    table += "\\usepackage[a4paper]{geometry}\\n"
    # table += "\\geometry{verbose, tmargin = 1.5cm, bmargin = 1.5cm, lmargin = 1cm, rmargin = 1cm}\\n"
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
            LO_NHO = [round(x, 2) for x in param_dict['LO']['NHO']]
            LO_HO = [round(x, 2) for x in param_dict['LO']['HO']]
            NLO_NHO = [round(x, 2) for x in param_dict['NLO']['NHO']]
            NLO_HO = [round(x, 2) for x in param_dict['NLO']['HO']]

            if model_row:

                table += f"{mod_dict[model_nr]} & {uv_param_dict[parameter]} & {LO_NHO} & {LO_HO} & {NLO_NHO} & {NLO_HO} \\\\\n"
                model_row = False
            else:
                table += f" & {uv_param_dict[parameter]} & {LO_NHO} & {LO_HO} & {NLO_NHO} & {NLO_HO}  \\\\\n"

    table += "\\hline\n"
    table += "\\end{tabular}\n"
    table += "\\caption{\label{demo-table}95\\% CL intervals on the heavy scalar fields UV couplings.}\n"
    table += "\\end{center}\n"
    table += "\\end{table}\n"
    table += "\\end{document}\n"

    return table


latex_code = nested_dict_to_latex_table(table_dict)

# Save LaTeX code to a .tex file
with open("table.tex", "w") as file:
    file.write(latex_code)

# Compile LaTeX code to PDF
subprocess.run(["pdflatex", "-interaction=batchmode", "table.tex"])
subprocess.run(["rm", "table.aux", "table.log"])
# Rename the output file

plot_nr = 1
n_rows = 5
n_cols = 3
fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))
for mod_nr in range(21):
    print(mod_nr)

    # path to posterior with model number
    posterior_path_mod = Path(posterior_path.format(mod_nr, 'LO', 'NHO'))
    if posterior_path_mod.exists():
        # Open the JSON file and load its contents
        with open(posterior_path_mod) as f:
            posterior = json.load(f)

        for key, samples in posterior.items():
            print(key)
            if key.startswith("O"):
                continue
            else:
                ax = plt.subplot(n_rows, n_cols, plot_nr)
                plot_nr += 1

                ax.hist(samples, bins='fd', density=True,
                            edgecolor="black",
                            alpha=0.4)

                ax.set_ylim(0, ax.get_ylim()[1] + ax.get_ylim()[1] * 0.2)

                x_pos = ax.get_xlim()[0] + 0.05 * (ax.get_xlim()[1] - ax.get_xlim()[0])
                y_pos = 0.95 * ax.get_ylim()[1]


                ax.tick_params(which="both", direction="in", labelsize=22.5)
                ax.tick_params(labelleft=False)

                ax.text(x_pos, y_pos, f"{uv_param_dict[key]}", fontsize=15, ha='left', va='top')


fig.savefig('./granada_posteriors.pdf')

subprocess.call("gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=scalar_models_granada.pdf table.pdf granada_posteriors.pdf", shell=True)

