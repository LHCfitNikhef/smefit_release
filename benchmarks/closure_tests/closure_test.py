# -*- coding: utf-8 -*-
# This script repeatedly generates independent pseudodata and records the chi2 at the best
# fit point. It belongs to Fig D.1 in the smefit3 paper.

import subprocess

projection_runcard = "./closure_hl_only.yaml"

fit_runcard = "./A_smefit_hllhc_only_glob_NLO_NHO.yaml"

# run fits repeatedly on different pseudo datasets
n_exp = 10
for i in range(n_exp):

    # run projection module
    subprocess.run(["smefit", "PROJ", "--closure", projection_runcard])

    # run analytic module without samples
    subprocess.run(["smefit", "A", fit_runcard])
