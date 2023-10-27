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
from sigfig import round

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc, use
from histogram_tools import find_xrange

from smefit.runner import Runner
from smefit.chi2 import Scanner

collections = ["OneLoop"]


here = pathlib.Path(__file__).parent

# result dir
result_dir = here / "results"
Path.mkdir(result_dir, parents=True, exist_ok=True)

mod_list = []
for col in collections:
    base_path = pathlib.Path(f"{here.parent}/runcards/uv_models/UV_scan/{col}/")
    sys.path = [str(base_path)] + sys.path
    for p in base_path.iterdir():
        if p.name.startswith("InvarsFit") and p.suffix == ".py":
            mod_list.append(importlib.import_module(f"{p.stem}"))


use("PDF")
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", **{"usetex": True, "latex.preamble": r"\usepackage{amssymb}"})

# compute the invariants
pQCDs = ['LO', 'NLO']
EFTs = ['NHO', 'HO']

for model in mod_list:
    for pQCD in pQCDs:
        for EFT in EFTs:
            print(model)
            model.MODEL_SPECS['pto'] = pQCD
            model.MODEL_SPECS['eft'] = EFT
            invariants = []
            for k, attr in model.__dict__.items():
                if k.startswith('inv'):
                    invariants.append(attr)
            try:
                model.inspect_model(model.MODEL_SPECS, invariants)
            except FileNotFoundError:
                continue

# Specify the path to the JSON file
posterior_path = f"{here.parent}/results/{{}}_UV_{{}}_{{}}_{{}}_NS/inv_posterior.json"


def nested_dict(num_levels):
    if num_levels == 1:
        return defaultdict(dict)
    else:
        return defaultdict(lambda: nested_dict(num_levels - 1))


def compute_bounds(collection, mod_nrs, label, caption):

    table = "\\begin{table}\n"
    table += "\\begin{center}\n"
    table += "\\scriptsize\n"
    table += "\\renewcommand{\\arraystretch}{1.5}"
    table += "\\begin{tabular}{c|c|c|c|c|c}\n"
    table += "Model & UV invariants & LO $\\mathcal{O}\\left(\\Lambda^{-2}\\right)$ &LO $\\mathcal{O}\\left(\\Lambda^{-4}\\right)$ & NLO $\\mathcal{O}\\left(\\Lambda^{-2}\\right)$ & NLO $\\mathcal{O}\\left(\\Lambda^{-4}\\right)$ \\\\\n"
    table += "\\toprule\n"

    table_dict = nested_dict(4)

    n_params = 0
    n_rows = 4
    n_cols = 3
    plot_nr = 1
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))

    for mod_nr in mod_nrs:
        if mod_nr == 5:
            continue
        print(mod_nr)

        model_row = True

        n_invariants = []
        ups = []
        lows = []
        for pQCD in ["LO"]:

            for EFT in ["NHO", "HO"]:
                n_inv = 0
                # path to posterior with model number
                posterior_path_mod = Path(posterior_path.format(collection, mod_nr, pQCD, EFT))

                if posterior_path_mod.exists():

                    # Open the JSON file and load its contents
                    with open(posterior_path_mod) as f:
                        posterior = json.load(f)

                    for key, samples in posterior.items():
                        if not key.startswith("inv"):

                            if not key.startswith('O'):
                                low, up = az.hdi(np.array(samples), .95)
                                lows.append(low)
                                ups.append(up)

                            continue

                                # # plot chi2_profile
                                # yaml_path = f"{here.parent}/runcards/{collection}_UV_{mod_nr}_{pQCD}_{EFT}_NS.yaml"
                                # fit_card = pathlib.Path(yaml_path)
                                # runner = Runner.from_file(fit_card.absolute())
                                # low, up = az.hdi(np.array(samples), .95)
                                # runner.run_card['coefficients'][key] = {'max': 1.2 * up, 'min': 0.8 * low}
                                # label = '{} {}'.format(EFT, pQCD)
                                #
                                # scan = Scanner(runner.run_card, n_replica=0)
                                # scan.compute_scan()
                                #
                                # for key, chi2 in scan.chi2_dict.items():
                                #     ax = plt.subplot(n_rows, n_cols, plot_nr)
                                #     ax.plot(chi2['x'], chi2[0], label=label)
                                #     ax.set_xlabel(uv_param_dict[key])
                                #     plt.legend()
                                #
                                # if EFT == 'HO':
                                #     plot_nr += 1

                        else:
                            # if min(samples) >= 0:
                            #     low = 0
                            #     up = np.nanpercentile(samples, 95.0)
                            #     up_str = "{}".format(round(up, sigfigs=4, notation='sci')).replace('E', 'e').replace('e0', '')
                            #     if 'e' in up_str:
                            #         base, exp = up_str.split('e')
                            #         up_str = "{}$\\mathrm{{e}}{{{}}}$".format(base, exp)
                            #     table_dict[mod_nr][key][pQCD][EFT] = up_str

                            # else:

                            low, up = az.hdi(np.array(samples), .95)


                            # low_str = "{}".format(round(low, sigfigs=4, notation='sci')).replace('E', 'e').replace(
                            #     'e0', '')
                            # up_str = "{}".format(round(up, sigfigs=4, notation='sci')).replace('E', 'e').replace(
                            #     'e0', '')
                            # if 'e' in low_str:
                            #     base, exp = low_str.split('e')
                            #     low_str = "{}$\\mathrm{{e}}{{{}}}$".format(base, exp)
                            # if 'e' in up_str:
                            #     base, exp = up_str.split('e')
                            #     up_str = "{}$\\mathrm{{e}}{{{}}}$".format(base, exp)
                            low_str = "{}".format(round(low, sigfigs=4))
                            up_str = "{}".format(round(up, sigfigs=4))
                            table_dict[mod_nr][key][pQCD][EFT] = "[{}, {}]".format(low_str, up_str)

                            # if up > 200 or low < -200:
                            #     table_dict[mod_nr][key][pQCD][EFT] = "$\\times$"

                            n_inv += 1
                    n_invariants.append(n_inv)

################

        # chi2  profile

        for pQCD in ["LO"]:

            chi2_mins = []
            for EFT in ["NHO", "HO"]:

                yaml_path = f"{here.parent}/runcards/{collection}_UV_{mod_nr}_{pQCD}_{EFT}_NS.yaml"
                fit_card = pathlib.Path(yaml_path)
                if fit_card.exists():
                    ax = plt.subplot(n_rows, n_cols, plot_nr)
                    runner = Runner.from_file(fit_card.absolute())
                else:
                    continue

                for key in runner.run_card['coefficients'].keys():
                    if key.startswith('O') and mod_nr != 3:
                        continue
                    delta = 0.3 * (max(ups) - min(lows))
                    if mod_nr == 3:
                        runner.run_card['coefficients'][key] = {'max':0.01, 'min': -0.05}
                    else:
                        runner.run_card['coefficients'][key] = {'max': max(ups) + delta, 'min': min(lows) - delta}
                order_EFT = -2 if EFT == 'NHO' else -4
                label = f"$\mathcal{{O}}\\left(\Lambda^{{{order_EFT}}}\\right)$"

                scan = Scanner(runner.run_card, n_replica=0)
                scan.compute_scan()


                for key, chi2 in scan.chi2_dict.items():
                    if mod_nr == 3:
                        if key != 'Oll':
                            continue
                    else:
                        if key.startswith('O'):
                            continue

                    color = 'C0' if EFT == 'NHO' else 'C1'

                    ax.plot(chi2['x'], chi2[0], label=label, color=color)
                    if mod_nr == 3:
                        ax.set_xlabel(inv_param_dict[mod_nr]['inv1'])
                    else:
                        ax.set_xlabel(uv_param_dict[key])
                    ax.set_ylabel(f"$\chi^2_{{\mathrm{{tot}}}} \\times n_{{\mathrm{{dat}}}}$")

                    chi2_min = np.array(chi2[0]).min()
                    chi2_mins.append(chi2_min)

                    ax.axhline(
                        y=chi2_min + 3.84, ls="dotted", color=color,
                    )

                plt.legend()
                ax.set_ylim(min(chi2_mins) - 0.2, max(chi2_mins) + 4.5)

                if EFT == 'HO':
                    plot_nr += 1




        if n_invariants:
            n_params += max(n_invariants)
        for parameter, param_dict in table_dict[mod_nr].items():

            LO_NHO = param_dict["LO"]["NHO"]
            LO_HO = param_dict["LO"]["HO"]
            NLO_NHO = param_dict["NLO"]["NHO"]
            NLO_HO = param_dict["NLO"]["HO"]
            if model_row:
                table += f"{mod_dict[mod_nr]} & {inv_param_dict[mod_nr][parameter]} & {LO_NHO} & {LO_HO} & {NLO_NHO} & {NLO_HO} \\\\\n"
                model_row = False
            else:
                table += f" & {inv_param_dict[mod_nr][parameter]} & {LO_NHO} & {LO_HO} & {NLO_NHO} & {NLO_HO}  \\\\\n"

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += f"\\caption{{{caption}}}\n"
    table += "\\end{center}\n"
    table += "\\end{table}\n"

    plt.tight_layout()
    fig.savefig('/data/theorie/jthoeve/smefit_release/smefit_uv_test/referee_report/chi2_scans/heavy_scalars/test.pdf')

    return table, n_params



scalar_mdl_nrs = range(21)
vboson_mdl_nrs = range(21, 37)
vfermion_mdl_nrs = range(37, 50)
mp_mdl_idx = ["Q1_Q7_W"]
oneloop_mdl_idx = ["Varphi"]
oneloop_mdl_idx_varm = ["Varphi_massbound"]

latex_table_scalar, n_scalars = compute_bounds("Granada", scalar_mdl_nrs, "cl-heavy-scalar",
                                              "95\\% CL intervals of the heavy scalar fields UV couplings.")
import pdb ;pdb.set_trace()

latex_table_vboson, n_vbosons = compute_bounds("Granada", vboson_mdl_nrs, "cl-heavy-vboson",
                                             "95\\% CL intervals of the heavy vector boson fields UV couplings.")

latex_table_vfermion, n_vfermions = compute_bounds("Granada", vfermion_mdl_nrs, "cl-heavy-vfermion",
                                                  "95\\% CL intervals of the heavy vector fermion fields UV couplings.")

latex_table_mp, n_mp = compute_bounds("Multiparticle", mp_mdl_idx, "cl-mp-model-spl", "95\\% CL intervals of the UV couplings that enter in the multiparticle model.")
#latex_table_1l, n_scalars_1L = compute_bounds("OneLoop", oneloop_mdl_idx, "cl-1l-model", "95\\% CL intervals of the UV couplings in \\varphi at one-loop.")
#latex_table_1l_varm, n_scalars_1L = compute_bounds("OneLoop", oneloop_mdl_idx_varm, "cl-1l-model", "95\\% CL intervals of the UV couplings in \\varphi at one-loop.")


#Save LaTeX code to a .tex file
# with open(result_dir / "table_scalar.tex", "w") as file:
#     file.write(latex_table_scalar)
with open(result_dir / "table_vboson.tex", "w") as file:
   file.write(latex_table_vboson)
# with open(result_dir / "table_vfermion.tex", "w") as file:
#     file.write(latex_table_vfermion)
# with open(result_dir / "table_mp.tex", "w") as file:
#    file.write(latex_table_mp)
# with open(result_dir / "table_1l.tex", "w") as file:
#     file.write(latex_table_1l)
# with open(result_dir / "table_1l_varm.tex", "w") as file:
#     file.write(latex_table_1l_varm)



def plot_uv_posterior(n_params, collection, mod_nrs, EFT=None, name=None, pQCD=None):
    plot_nr = 1

    # Calculate number of rows and columns
    n_cols = max(int(np.ceil(np.sqrt(n_params))), 4)
    if n_params < n_cols:
        n_cols = n_params

    n_rows = int(np.ceil(n_params / n_cols))

    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))

    for mod_nr in mod_nrs:
        if mod_nr == 5:
            continue
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
            try:
                with open(posterior_path_mod_1) as f:
                    posterior_1 = json.load(f)

                with open(posterior_path_mod_2) as f:
                    posterior_2 = json.load(f)
            except FileNotFoundError:
                continue

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

                    x_low, x_high = find_xrange([samples_1, samples_2], 0.05)


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

                    ax.tick_params(which="both", direction="in", labelsize=22.5)
                    ax.tick_params(labelleft=False)

                    ax.text(
                        0.05,
                        0.95,
                        inv_param_dict[mod_nr][key],
                        fontsize=15,
                        ha="left",
                        va="top",
                        transform=ax.transAxes
                    )

                    ax.text(
                        0.95,
                        0.95,
                        mod_dict[mod_nr],
                        fontsize=15,
                        ha="right",
                        va="top",
                        transform=ax.transAxes
                    )


    if pQCD is None:
        order_EFT = -2 if EFT == 'NHO' else -4
        fig.legend([f"$\mathrm{{LO}}\;\mathcal{{O}}\\left(\Lambda^{{{order_EFT}}}\\right)$",
                    f"$\mathrm{{NLO}}\;\mathcal{{O}}\\left(\Lambda^{{{order_EFT}}}\\right)$"], loc="upper center",
                   ncol=2,
                   prop={"size": 25 * (n_cols * 4) / 20}, bbox_to_anchor=(0.5, 1.0), frameon=False)

        if n_rows > 1:
            plt.tight_layout(rect=[0, 0.05 * (5. / n_rows), 1, 1 - 0.05 * (5. / n_rows)])  # make room for the legend
        else:
            plt.tight_layout(rect=[0, 0.08, 1, 0.92])  # make room for the legend
        if name is not None:
            fig.savefig(result_dir / "{}_posteriors_LO_vs_NLO_{}_{}.pdf".format(collection, EFT, name))
        else:
            fig.savefig(result_dir / "{}_posteriors_LO_vs_NLO_{}.pdf".format(collection, EFT))
    elif EFT is None:
        fig.legend([f"$\mathrm{{{pQCD}}}\;\mathcal{{O}}\\left(\Lambda^{{-2}}\\right)$",
                    f"$\mathrm{{{pQCD}}}\;\mathcal{{O}}\\left(\Lambda^{{-4}}\\right)$"], ncol=2,
                   prop={"size": 25 * (n_cols * 4) / 20}, bbox_to_anchor=(0.5, 1.0), loc='upper center', frameon=False)
        if n_rows > 1:
            plt.tight_layout(rect=[0, 0.05 * (5. / n_rows), 1, 1 - 0.05 * (5. / n_rows)])  # make room for the legend
        else:
            plt.tight_layout(rect=[0, 0.08, 1, 0.92])  # make room for the legend

        if name is not None:
            fig.savefig(result_dir / "{}_posteriors_HO_vs_NHO_{}_{}.pdf".format(collection, pQCD, name))
        else:
            fig.savefig(result_dir / "{}_posteriors_HO_vs_NHO_{}.pdf".format(collection, pQCD))


# plot_uv_posterior(n_mp, "Multiparticle", mp_mdl_idx, EFT="NHO")
# plot_uv_posterior(n_mp, "Multiparticle", mp_mdl_idx, EFT="HO")
# plot_uv_posterior(n_mp, "Multiparticle",mp_mdl_idx, pQCD="LO")
# plot_uv_posterior(n_mp, "Multiparticle", mp_mdl_idx, pQCD="NLO")
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
# plot_uv_posterior(n_scalars, "Granada", scalar_mdl_nrs, name="scalars", pQCD="LO")
# plot_uv_posterior(n_scalars, "Granada", scalar_mdl_nrs, name="scalars", pQCD="NLO")

n_scalars_1L = 2
plot_uv_posterior(n_scalars_1L, "OneLoop", oneloop_mdl_idx, EFT="NHO")
plot_uv_posterior(n_scalars_1L, "OneLoop", oneloop_mdl_idx, EFT="HO")
plot_uv_posterior(n_scalars_1L, "OneLoop", oneloop_mdl_idx, pQCD="LO")
plot_uv_posterior(n_scalars_1L, "OneLoop", oneloop_mdl_idx, pQCD="NLO")


for file in result_dir.iterdir():
    if file.suffix == ".pdf":
        subprocess.run(["pdfcrop", str(file), str(file)])
