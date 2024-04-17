import numpy as np

def make_eos_data_2023_01():

    # do something with flavio here
    n_dat = 5

    data = np.ones(n_dat)
    sigma = 0.1
    theory_sm = np.random.normal(data, sigma)

    # inject anomaly in the first data point
    data[0] = theory_sm[0] + sigma
    cov = np.diag(sigma ** 2 * np.ones(n_dat))
    invcovmat = np.linalg.inv(cov)

    return lambda wc: np.einsum("i,ij,j->", data - theory_sm - 1e-5 * wc, invcovmat, data - theory_sm - 1e-5 * wc)


dataset_dict = {
    'EOS-DATA-2023-01': (
        make_eos_data_2023_01,
        ['ubenue::cVL', 'ubenue::cVR', 'ubenue::cSL', 'ubenue::cSR', 'ubenue::cT']
    )
}
