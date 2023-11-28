import numpy as np

def make_eos_data_2023_01():

    # do something with flavio here
    data = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    cov = np.diag([0.1, 0.1, 0.1, 0.1, 0.1])
    invcovmat = np.linalg.inv(cov)
    return lambda wc: np.einsum("i,ij,j->", data - wc, invcovmat, data - wc)


dataset_dict = {
    'EOS-DATA-2023-01': (
        make_eos_data_2023_01,
        ['ubenue::cVL', 'ubenue::cVR', 'ubenue::cSL', 'ubenue::cSR', 'ubenue::cT']
    )
}
