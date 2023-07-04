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


def nested_dict(num_levels):
    if num_levels == 1:
        return defaultdict(dict)
    else:
        return defaultdict(lambda: nested_dict(num_levels - 1))

table_dict = nested_dict(4)

for mod_nr in range(49):
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
    table += "\\geometry{verbose, tmargin = 1.5cm, bmargin = 1.5cm, lmargin = 1cm, rmargin = 1cm}\\n"
    table += "\\begin{document}\n"
    table += "\\begin{tabular}{|c|c|c|c|c|c|}\n"
    table += "\\hline\n"
    table += "Model & Parameter & LO lin & LO quad & NLO lin & NLO quad \\\\\n"
    table += "\\hline\n"

    for model_nr, params in nested_dict.items():
        model_row = True
        for parameter, param_dict in params.items():
            LO_NHO = [round(x, 2) for x in param_dict['LO']['NHO']]
            LO_HO = [round(x, 2) for x in param_dict['LO']['HO']]
            NLO_NHO = [round(x, 2) for x in param_dict['NLO']['NHO']]
            NLO_HO = [round(x, 2) for x in param_dict['NLO']['HO']]

            if model_row:

                table += f"{model_nr} & {parameter} & {LO_NHO} & {LO_HO} & {NLO_NHO} & {NLO_HO} \\\\\n"
                model_row = False
            else:
                table += f" & {parameter} & {LO_NHO} & {LO_HO} & {NLO_NHO} & {NLO_HO}  \\\\\n"

    table += "\\hline\n"
    table += "\\end{tabular}"
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

print("PDF file saved as 'output.pdf'.")

