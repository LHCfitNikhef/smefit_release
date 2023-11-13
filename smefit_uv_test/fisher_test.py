import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pymultinest.solve import solve
import arviz as az


def prior_uv(hypercube):
    """
    Update the prior function

    Parameters
    ----------
        hypercube : np.ndarray
            hypercube prior

    Returns
    -------
        flat_prior : np.ndarray
            updated hypercube prior
    """

    return hypercube


def prior(hypercube):
    """
    Update the prior function

    Parameters
    ----------
        hypercube : np.ndarray
            hypercube prior

    Returns
    -------
        flat_prior : np.ndarray
            updated hypercube prior
    """
    max = 20
    min = - 20
    return hypercube * (max - min) + min


def matching_1(g):
    g1, g2 = g
    return g1 ** 2


def matching_2(g):
    g1, g2 = g
    return g1 * g2


def chi2(g):
    chi_sq = 0

    # bin 1
    sigma_exp = 1
    sigma_sm = sigma_exp
    c1 = matching_1(g)
    c2 = matching_2(g)
    sigma_1 = -10
    sigma_2 = 50
    sigma_th = sigma_sm + c1 * sigma_1 + c2 * sigma_2
    chi_sq += (sigma_exp - sigma_th) ** 2

    # bin 1
    # sigma_exp = 1
    # sigma_sm = sigma_exp
    # c1 = matching_1(g)
    # c2 = matching_2(g)
    # sigma_1 = 0.2
    # sigma_2 = -0.1
    # sigma_th = sigma_sm + c1 * sigma_1 + c2 * sigma_2
    # chi_sq += (sigma_exp - sigma_th) ** 2

    return chi_sq


def log_like(g):
    return -0.5 * chi2(g)


def fisher(g1, g2):
    f = np.zeros((g1.shape[0], 2, 2))
    g = np.zeros((g1.shape[0], 2, 2))
    for i, (g1_i, g2_i) in enumerate(zip(g1, g2)):
        c1 = matching_1([g1_i, g2_i])
        c2 = matching_2([g1_i, g2_i])

        f[i, 0, 0] = 400 * g1_i ** 2 - 2000 * g1_i * g2_i + 2500 * g2_i ** 2
        f[i, 0, 1] = -1000 * g1_i ** 2 + 2500 * g1_i * g2_i
        f[i, 1, 0] = f[i, 0, 1]
        f[i, 1, 1] = 2500 * g1_i ** 2

        g[i, 0, 0] = (-60 * (30 * c1 - 50 * c2))
        g[i, 0, 1] = (50 * (30 * c1 - 50 * c2))
        g[i, 1, 0] = g[i, 0, 1]
        g[i, 1, 1] = 0

    h = f - g
    
    return f


res = solve(log_like, prior, n_dims=2, n_params=2, n_live_points=2000)

g1_samples = res['samples'][:, 0]
g2_samples = res['samples'][:, 1]
c1_samples = matching_1(res['samples'].T)
c2_samples = matching_2(res['samples'].T)

f = fisher(g1_samples, g2_samples)
f_avg = f.mean(axis=0)
eigval, eigvec = np.linalg.eig(f_avg)


def find_xrange(samples):
    x_low = []
    x_high = []

    for sample in samples:

        n, bins = np.histogram(
            sample,
            bins="fd",
            density=True,

        )

        max_index = np.argmax(n)
        threshold = 0.05 * n[max_index]
        indices = np.where(n >= threshold)[0]

        low, up = az.hdi(sample, hdi_prob=.95)

        index_max = np.max(indices)
        index_min = np.min(indices)

        x_high.append(bins[index_max])

        if low < 0:
            x_low.append(bins[index_min])
        else:
            x_low.append(- 0.1 * bins[index_max])

    return min(x_low), max(x_high)


# n_rows = 4
# n_cols = 2
# fig, ax = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
#
# x_low_g1, x_high_g1 = find_xrange([g1_samples])
# x_low_g2, x_high_g2 = find_xrange([g2_samples])
# x_low_c1, x_high_c1 = find_xrange([c1_samples])
# x_low_c2, x_high_c2 = find_xrange([c2_samples])
#
# ax[0, 0].hist(g1_samples[(g1_samples > x_low_g1) & (g1_samples < x_high_g1)], bins='fd', alpha=.4, density=True,
#               edgecolor='black')
# ax[0, 0].set_xlabel('$g_1$')
#
# ax[0, 1].hist(g2_samples[(g2_samples > x_low_g2) & (g2_samples < x_high_g2)], bins='fd', alpha=.4, density=True,
#               edgecolor='black')
# ax[0, 1].set_xlabel('$g_2$')
#
# ax[1, 0].hist(c1_samples[c1_samples < 0.05 ** 2], bins='fd', alpha=.4, density=True, edgecolor='black')
# ax[1, 0].set_xlabel('$c_1$')
#
# ax[1, 1].hist(c2_samples[(c2_samples > x_low_c2) & (c2_samples < x_high_c2)], bins='fd', alpha=.4, density=True,
#               edgecolor='black')
# ax[1, 1].set_xlabel('$c_2$')
#
# ax[2, 0].scatter(res['samples'][:, 0], res['samples'][:, 1], s=1)
# ax[2, 0].set_xlabel('$g_1$')
# ax[2, 0].set_ylabel('$g_2$')
# # fig_width, fig_height = ax[2, 0].get_size_inches()
#
# ax[2, 0].arrow(0, 0, eigvec[0, 0], eigvec[1, 0], color='red', head_width=.5)
# ax[2, 0].arrow(0, 0, eigvec[0, 1], eigvec[1, 1], color='red', head_width=.5)
# # ax[2, 0].arrow([0, eigvec[0, 1]], [0, eigvec[1, 1]], color='red')
# ax[2, 0].set_xlim(-5, 5)
# ax[2, 0].set_ylim(-5, 5)
#
# ax[3, 0].hist(res['samples'] @ eigvec[:, 0], bins='fd', alpha=.4, density=True, edgecolor='black')
# ax[3, 0].set_xlabel('$v_1$')
# ax[3, 1].hist(res['samples'] @ eigvec[:, 1], bins='fd', alpha=.4, density=True, edgecolor='black')
# ax[3, 1].set_xlabel('$v_2$')

n_rows = 1
n_cols = 3
fig, ax = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

x_low_g1, x_high_g1 = find_xrange([g1_samples])
x_low_g2, x_high_g2 = find_xrange([g2_samples])
x_low_c1, x_high_c1 = find_xrange([c1_samples])
x_low_c2, x_high_c2 = find_xrange([c2_samples])

ax[0].scatter(res['samples'][:, 0], res['samples'][:, 1], s=1)
ax[0].arrow(0, 0, eigvec[0, 0], eigvec[1, 0], color='red', head_width=.1)
ax[0].arrow(0, 0, eigvec[0, 1], eigvec[1, 1], color='red', head_width=.1)
ax[0].set_xlim(-2, 2)
ax[0].set_ylim(-2, 2)
ax[0].set_xlabel('$g_1$')
ax[0].set_ylabel('$g_2$')

ax[1].hist(res['samples'] @ eigvec[:, 0], bins='fd', alpha=.4, density=True, edgecolor='black')
ax[1].set_xlabel('$v_1$')
ax[2].hist(res['samples'] @ eigvec[:, 1], bins='fd', alpha=.4, density=True, edgecolor='black')
ax[2].set_xlabel('$v_2$')

plt.tight_layout()
fig.savefig('./fisher_test.pdf')
import pdb; pdb.set_trace()
