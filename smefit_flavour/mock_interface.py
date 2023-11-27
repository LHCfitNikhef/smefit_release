import numpy as np
import wilson
import matplotlib.pyplot as plt
import scipy
import pathlib

from smefit.runner import Runner
from smefit.chi2 import Scanner
import smefit.log as log
from smefit.log import print_banner, setup_console
from smefit.optimize.ultranest import USOptimizer


def make_eos_data_2023_01():
    mean = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    sigma = np.diag([0.1, 0.1, 0.1, 0.1, 0.1])
    sigma_inv = np.linalg.inv(sigma)

    return lambda wc: np.dot((wc - mean),np.dot(sigma_inv, (wc - mean)))

def make_smeft_likelihood(name:str, smeft_ops:list[str]):

    LIKELIHOODS = {
        'EOS-DATA-2023-01': (
            make_eos_data_2023_01,
            [ 'ubenue::cVL', 'ubenue::cVR', 'ubenue::cSL', 'ubenue::cSR', 'ubenue::cT' ]
        )
    }

    if name not in LIKELIHOODS.keys():
        return None

    factory, wet_ops = LIKELIHOODS[name]

    # ideally, we proceed in three step as follows:
    #   1. compute SMEFT running matrix from mu_0 to mu_m, the matching scale
    #   2. compute SMEFT-WET matching matrix at mu_m
    #   3. compute WET running matrix from mu_m to mu_WET
    # instead, we take a shortcut but using wilson.Wilson.match_run. This can lead
    # to excluding some contributions from SMEFT operators induced by the SMEFT-running.
    # 1.a determine which SMEFT WC are induced by running
    M_smeft_wet = []
    for i in range(len(smeft_ops)):
        smeft_values = { name: 0.0 for name in smeft_ops }
        smeft_values[smeft_ops[i]] = 1.0
        smeft_wc = wilson.Wilson(smeft_values, 1000, 'SMEFT', 'Warsaw')
        wet_wc = smeft_wc.match_run(4.2, 'WET', 'EOS')
        M_smeft_wet.append([np.real(wet_wc.dict[wet_op]) if wet_op in wet_wc.dict.keys() else 0.0 for wet_op in wet_ops])

    M_smeft_wet = np.array(M_smeft_wet).T

    wet_llh = factory()

    return lambda wc: wet_llh(np.dot(M_smeft_wet, wc))

smeft_ops = ['lq3_1111', 'lq3_1133']
smeft_llh = make_smeft_likelihood('EOS-DATA-2023-01', smeft_ops)

log_path = pathlib.Path('/data/theorie/jthoeve/smefit_release/cluster/logs/flavour_log.log')
fit_card = pathlib.Path("/data/theorie/jthoeve/smefit_release/runcards/NS_smefit_flavour_LO_HO.yaml")

setup_console(log_path)
print_banner()
runner = Runner.from_file(fit_card.absolute())
log.console.log("Running : Nested Sampling Fit ")

# set up the optimizer
opt = USOptimizer.from_dict(runner.run_card)
opt.external_chi2 = smeft_llh

import pdb; pdb.set_trace()
# start fit
opt.run_sampling()
import pdb; pdb.set_trace()
#runner.ultranest(runner.run_card)



result = scipy.optimize.minimize(smeft_llh, [0.001, -0.01])
