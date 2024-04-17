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


class EOSLikelihood:
    
    """
    Constructs the WET likelihood as a function of the SMEFT operators, which are first run down to the WET
    scale and then used to evaluate the WET likelihoods

    Parameters
    ----------
    flavour_datasets: list[str]
        passed as string
    smeft_ops: list[str]


    Returns
    -------
    function: A lambda function that return the overall likelihood for all datasets in `flavour_datasets`
    """

    def __init__(self, coefficients):

        self.smeft_ops = ['phiq3_33', 'uG_33']  # List of SMEFT operators
        self.flavour_datasets = ['EOS-DATA-2023-01']  # List of flavour datasets

        # do the running once
        self.wet_llhs, self.m_smeft_wets = self.compute_rg()

        # TODO: split up functionality
        # self.wet_llhs = self.get_wet_llhs()
        # self.m_smeft_wets = self.compute_rg()


    def compute_rg(self):
        """
        Computes the SMEFT to WET RG matrix
        """

        wet_llhs = []
        m_smeft_wets = []
        for dataset_name in self.flavour_datasets:

            if dataset_name not in fl_llh.dataset_dict.keys():
                continue

            wet_llh_factory, wet_ops = fl_llh.dataset_dict[dataset_name]

            # compute running matrix from the SMEFT to the WET
            smeft_value_init = 1e-4
            m_smeft_wet = []
            for smeft_op in self.smeft_ops:
                smeft_values = {name: 0.0 for name in self.smeft_ops}
                smeft_values[smeft_op] = smeft_value_init
                smeft_wc = wilson.Wilson(smeft_values, 1e3, 'SMEFT', 'Warsaw')
                smeft_wc.set_option('smeft_accuracy', 'leadinglog')

                wet_wc = smeft_wc.match_run(scale=4.2, eft='WET', basis='EOS')
                m_smeft_wet.append(
                    [np.real(wet_wc.dict[wet_op]) if wet_op in wet_wc.dict.keys() else 0.0 for wet_op in wet_ops])

            m_smeft_wet = np.array(m_smeft_wet).T / smeft_value_init

            m_smeft_wets.append(m_smeft_wet)
            wet_llhs.append(wet_llh_factory())

        return wet_llhs, m_smeft_wets

    def compute_chi2(self, coefficient_values):

        chi2 = sum(-0.5 * wet_llh(np.dot(self.m_smeft_wets[0], coefficient_values)) for wet_llh in self.wet_llhs)
        return chi2




