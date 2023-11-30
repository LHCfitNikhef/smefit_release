import numpy as np
import wilson
# import flavio
import matplotlib.pyplot as plt
import scipy
import pathlib

import flavour_likelihoods as fl_llh

from smefit.runner import Runner
from smefit.chi2 import Scanner
import smefit.log as log
from smefit.log import print_banner, setup_console
from smefit.optimize.ultranest import USOptimizer


def make_flavour_chi2(flavour_datasets: list[str], smeft_ops: list[str]):
    """
    Constructs the WET likelihood as a function of the SMEFT operators, which are first run down to the WET
    scale and then used to evaluate the WET likelihoods

    Parameters
    ----------
    flavour_datasets: list[str]
        List of flavour datasets passed as string
    smeft_ops: list[str]
        List of SMEFT operators

    Returns
    -------
    function: A lambda function that return the overall likelihood for all datasets in `flavour_datasets`
    """

    wet_llhs = []
    for dataset_name in flavour_datasets:

        if dataset_name not in fl_llh.dataset_dict.keys():
            continue

        wet_llh_factory, wet_ops = fl_llh.dataset_dict[dataset_name]

        # compute running matrix from the SMEFT to the WET
        m_smeft_wet = []
        for smeft_op in smeft_ops:
            smeft_values = {name: 0.0 for name in smeft_ops}
            smeft_values[smeft_op] = 1
            smeft_wc = wilson.Wilson(smeft_values, 1e3, 'SMEFT', 'Warsaw')
            wet_wc = smeft_wc.match_run(scale=4.2, eft='WET', basis='EOS')
            m_smeft_wet.append(
                [np.real(wet_wc.dict[wet_op]) if wet_op in wet_wc.dict.keys() else 0.0 for wet_op in wet_ops])
        m_smeft_wet = np.array(m_smeft_wet).T

        wet_llhs.append(wet_llh_factory())

    return lambda wc: sum(-0.5 * wet_llh(np.dot(m_smeft_wet, wc)) for wet_llh in wet_llhs)


smeft_ops = ['lq3_1111', 'lq3_1133']
flavour_datasets = ['EOS-DATA-2023-01']

# construct the flavour likelihood as a function of the SMEFT operators
flavour_chi2 = make_flavour_chi2(flavour_datasets, smeft_ops)

