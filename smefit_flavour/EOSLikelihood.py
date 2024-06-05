# -*- coding: utf-8 -*-
import pathlib
import re

import flavour_likelihoods as fl_llh

# import flavio
import matplotlib.pyplot as plt
import numpy as np
import scipy
import wilson
import eos

import smefit.log as log
from smefit.chi2 import Scanner
from smefit.log import print_banner, setup_console
from smefit.optimize.ultranest import USOptimizer
from smefit.runner import Runner


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

    def __init__(self, coefficients, id, likelihood):

        self.id = id
        self.likelihood = likelihood

        # TODO: make path relative, not hardcoded
        eos_dataset = eos.DataSets(storage_directory='/data/theorie/jthoeve/smefit_release/smefit_flavour/datasets')

        # uncomment to download dataset, make automatic
        #eos_dataset.download(id)

        # TODO: 3rd return value is still None
        self.varied_parameters, self.neg_log_pdf, self.chi2 = eos_dataset.likelihood(id, likelihood)

        self.wet_ops = [p["name"] for p in self.varied_parameters] # WET operators

        # optimise: use 2 * self.neg_log_pdf
        # goodness of fit: use self.chi2

        # here we need Mila's program to determine which SMEFT operators enter under our flavour assumptions
        self.smeft_ops = ["phiq3_33", "uG_33"]  # SMEFT operators

        # do the running once
        self.compute_rg()

        # TODO: split up functionality
        # self.wet_llhs = self.get_wet_llhs()
        # self.m_smeft_wets = self.compute_rg()

    def compute_rg(self):
        """
        Computes the SMEFT to WET RG matrix
        """

        # compute running matrix from the SMEFT to the WET
        smeft_value_init = 1e-4
        m_smeft_wet = []
        for smeft_op in self.smeft_ops:
            smeft_values = {name: 0.0 for name in self.smeft_ops}
            smeft_values[smeft_op] = smeft_value_init
            smeft_wc = wilson.Wilson(smeft_values, 1e3, "SMEFT", "Warsaw")
            smeft_wc.set_option("smeft_accuracy", "leadinglog")

            wet_wc = smeft_wc.match_run(scale=4.2, eft="WET", basis="EOS")

            # the wcxf in the EOS basis sometimes includes Re in the key, sometimes not, how to treat this consistently?
            # working temp soln below
            m_smeft_wet_row = []
            for wet_op in self.wet_ops:
                if wet_op in wet_wc.dict.keys():
                    m_smeft_wet_row.append(np.real(wet_wc.dict[wet_op]))
                else:
                    wet_op_woRe = re.sub(r'Re\{(.*?)\}', r'\1', wet_op)
                    if wet_op_woRe in wet_wc.dict.keys():
                        m_smeft_wet_row.append(np.real(wet_wc.dict[wet_op_woRe]))
                    else:
                        m_smeft_wet_row.append(0.0)
            m_smeft_wet.append(m_smeft_wet_row)
            # m_smeft_wet.append(
            #     [
            #         np.real(wet_wc.dict[wet_op])
            #         if wet_op in wet_wc.dict.keys()
            #         else 0.0
            #         for wet_op in self.wet_ops
            #     ]
            # )

        m_smeft_wet = np.array(m_smeft_wet).T / smeft_value_init # maps from SMEFT to the WET
        self.m_smeft_wet = m_smeft_wet

    def compute_neg_log_likelihood(self, coefficient_values):
        wet_values = self.m_smeft_wet @ coefficient_values
        return 2 * self.neg_log_pdf(wet_values)

    # for the test statistic
    def compute_chi2(self, coefficient_values):
        neg_log_likelihood = self.compute_neg_log_likelihood(coefficient_values)
        return self.chi2(neg_log_likelihood)

    # def compute_chi2(self, coefficient_values):
    #
    #     chi2 = sum(-0.5 * wet_llh(np.dot(self.m_smeft_wets[0], coefficient_values)) for wet_llh in self.wet_llhs)
    #     return chi2
