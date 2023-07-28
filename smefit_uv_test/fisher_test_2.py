import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pymultinest.solve import solve
import arviz as az



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
    return g2 * g1


def chi2(g):
    chi_sq = 0

    # bin 1
    sigma_exp = 1
    sigma_sm = 1
    c1 = matching_1(g)
    c2 = matching_2(g)
    sigma_1 = -30
    sigma_2 = 50
    sigma_th = sigma_sm + c1 * sigma_1 + c2 * sigma_2
    chi_sq += (sigma_exp - sigma_th) ** 2

    return chi_sq


def log_like(g):
    return -0.5 * chi2(g)


def fisher(g1, g2):
    f = np.zeros((2, 2))

    f[0, 0] = 5400 * g1 ** 2 - 9000 * g1 *  g2 + 2500 * g2 ** 2
    f[0, 1] = -4500 * g1 ** 2 + 5000 * g1 * g2
    f[1, 0] = f[0, 1]
    f[1, 1] = 2500 * g1 ** 2

    eigval, eigvec = np.linalg.eig(f)

    return eigval, eigvec


res = solve(log_like, prior, n_dims=2, n_params=2, n_live_points=1000)

g1_samples = res['samples'][:, 0]
g2_samples = res['samples'][:, 1]
c1_samples = matching_1(res['samples'].T)
c2_samples = matching_2(res['samples'].T)

eigval_1, eigvec_1 = fisher(-0.5, -0.308)
eigval_2, eigvec_2 = fisher(0, 0.6)

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


n_rows = 1
n_cols = 1
fig, ax = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
#
# v0_samples = res['samples'] @ eigvec[:, 0]
# v1_samples = res['samples'] @ eigvec[:, 1]

# x_low_v0, x_high_v0 = find_xrange([v0_samples])
# x_low_v1, x_high_v1 = find_xrange([v1_samples])

g1 = np.linspace(-2, -0.001, 100)
ax.scatter(res['samples'][:, 0], res['samples'][:, 1], s=1)

ax.plot(g1, - 0.02 * (-0.2 - 30 * g1 ** 2) / g1, color='red')
ax.plot(-g1, - 0.02 * (-0.2 - 30 * g1 ** 2) / (-g1), color='red')
ax.arrow(-0.5, -0.308, 0.5 * eigvec_1[0, 0], 0.5 * eigvec_1[1, 0], color='k', head_width=.1, zorder=2)
ax.arrow(-0.5, -0.308, 0.5 * eigvec_1[0, 1], 0.5 * eigvec_1[1, 1], color='k', head_width=.1,zorder=2)
ax.arrow(0, 0.6, 0.5 * eigvec_2[0, 0], 0.5 * eigvec_2[1, 0], color='k', head_width=.1,zorder=2)
ax.arrow(0, 0.6, 0.5 * eigvec_2[0, 1], 0.5 * eigvec_2[1, 1], color='k', head_width=.1,zorder=2)


ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel('$g_1$')
ax.set_ylabel('$g_2$')



# ax[1].hist(v0_samples[(v0_samples > x_low_v0) & (v0_samples < x_high_v0)], bins='fd', alpha=.4, density=True, edgecolor='black')
# ax[1].set_xlabel('$v_0$')
# ax[2].hist(v1_samples[(v1_samples > x_low_v1) & (v1_samples < x_high_v1)], bins='fd', alpha=.4, density=True, edgecolor='black')
# ax[2].set_xlabel('$v_1$')

plt.tight_layout()

fig.savefig('./fisher_test_2.pdf')

