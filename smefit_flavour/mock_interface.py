import numpy as np
import wilson
import matplotlib.pyplot as plt
import scipy

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

result = scipy.optimize.minimize(smeft_llh, [0.001, -0.01])
