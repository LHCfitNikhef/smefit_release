# -*- coding: utf-8 -*-
import subprocess
from os import listdir


def latex_packages():

    L = [
        r"\documentclass{article}",
        r"\usepackage{float}",
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\usepackage{booktabs}",
        r"\usepackage[a4paper]{geometry}",
        r"\usepackage{array}",
        r"\usepackage{hyperref}",
        r"\usepackage{xcolor}",
        r"\usepackage{multirow}",
        r"\usepackage{pdflscape}",
        r"\allowdisplaybreaks",
        r"\newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}",
        r"\usepackage{graphicx}",
        r"\usepackage{tabularx}",
        r"\geometry{verbose, tmargin = 1.5cm, bmargin = 1.5cm, lmargin = 1cm, rmargin = 1cm}",
    ]
    return L


def multicolum_table_header(fit_labels, ncolumn=2):
    """Append the multicolumn table header"""
    L = [r"\hline"]
    temp = " & "
    for label in fit_labels:
        temp += r" & \multicolumn{%d}{c|}{%s} " % (ncolumn, label)
    temp += r" \\ \hline"
    L.append(temp)
    return L


def run_pdflatex(report, L, filename):
    """
    Dump to file and run pdflatex

    Parameters
    ----------
        report: str
            report name
        L : list(str)
            latex lines
        filename : str
            file name
    """

    L.append(r"\end{document}")
    L = [l + "\n" for l in L]

    latex_src = f"{report}/{filename}.tex"
    with open(latex_src, "w") as file:
        file.writelines(L)
    file.close()
    subprocess.call(
        f"pdflatex -halt-on-error -output-directory {report}/ {latex_src}", shell=True
    )
    subprocess.call(f"rm {report}/*.log {report}/*.aux {report}/*.out", shell=True)


def move_to_meta(report, filename):
    """
    Move pdf files to meta folder

    Parameters
    ----------
        report: str
            report name
        filename : str
            file names to be moved
    """
    subprocess.call(f"mkdir -p {report}/meta", shell=True)
    subprocess.call(f"mv {report}/{filename}.pdf {report}/meta/.", shell=True)


def chi2table_header(L, fit_labels):
    L.append(r"\hline")
    temp = r" \multicolumn{2}{|c|}{} & SM"
    for label in fit_labels:
        temp += f"& {label}"
    temp += r"\\ \hline"
    L.append(temp)
    L.append(
        r"Process & $N_{\rm data}$ & $\chi^ 2/N_{\rm data}$"
        + r"& $\chi^ 2/N_{data}$" * len(fit_labels)
        + r"\\ \hline"
    )
    return L
