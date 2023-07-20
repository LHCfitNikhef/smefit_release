import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import rc, use
import pandas as pd

use("PDF")
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], 'size': 20})
rc("text", **{"usetex": True, "latex.preamble": r"\usepackage{amssymb}"})

# Data for the comparison
categories = ['$S$', '$\\varphi$', '$\\Xi$', '$B_1$', '$W_1$', '$N$', '$E$', '$\\Delta_1$', '$\\Delta_3$', '$\\Sigma$',
              '$\\Sigma_1$', '$U$', '$D$', '$Q_5$', '$Q_7$', '$T_1$', '$T_2$', '$T$', '$Q_{17}$']

fitmaker_bounds = np.array(
    [1.7, 0.995, 1.1e-2, 6.9e-3, 8.6e-2, 3.8e-2, 2.2e-2, 1.7e-2, 2.9e-2, 4.5e-2, 2.7e-2, 7.2e-2, 3.8e-2, 0.24, 0.14,
     0.22, 0.099, 0.04, 0.88]) ** 0.5

smefit_ids = np.array([2, 5, 6, 22, 24, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 52])
guv = [['kS'], ['cotBeta', 'Z6'], ['kXi'], ['gB1H'], ['gW1H'], ['lamNef1'], ['lamEff1'], ['lamDelta1f1'], ['lamDelta3f1'],
       ['lamSigmaf1'], ['lamSigma1f1'], ['lamUf1'], ['lamDff1'], ['lamQ5f1'], ['lamQ7f1'], ['lamT1f1'], ['lamT2f1'],
       ['sLt'], ['lambdaQ']]

def compute_bounds(path_to_post):

    smefit_bounds = []
    for i, smefit_id in enumerate(smefit_ids):

        with open(path_to_post.format(smefit_id), encoding="utf-8") as f:
            posterior = json.load(f)
            if len(guv[i]) == 1:
                posterior_uv = posterior[guv[i][0]]
                smefit_bounds.append(np.percentile(np.abs(posterior_uv), 95))
            else:
                posterior_uv = np.array(posterior[guv[i][0]]) * np.array(posterior[guv[i][1]])
                smefit_bounds.append(np.percentile(np.abs(posterior_uv), 95))

    return smefit_bounds


post_smefit_nlo_nho = '/data/theorie/jthoeve/smefit_release/results/Fitmaker_UV_{}_NLO_NHO_NS/posterior.json'
post_smefit_nlo_ho = '/data/theorie/jthoeve/smefit_release/results/Fitmaker_UV_{}_NLO_HO_NS/posterior.json'
post_smefit_lo_nho = '/data/theorie/jthoeve/smefit_release/results/Fitmaker_UV_{}_LO_NHO_NS/posterior.json'
post_smefit_lo_ho = '/data/theorie/jthoeve/smefit_release/results/Fitmaker_UV_{}_LO_HO_NS/posterior.json'


smefit_bounds_nlo_nho = compute_bounds(post_smefit_nlo_nho)
smefit_bounds_nlo_ho = compute_bounds(post_smefit_nlo_ho)
smefit_bounds_lo_nho = compute_bounds(post_smefit_lo_nho)
smefit_bounds_lo_ho = compute_bounds(post_smefit_lo_ho)

# Set the total width of the bars
total_width = 6

# Set the width of each individual bar
bar_width = total_width / 5
x = np.arange(0, 6 * bar_width * len(categories), 6 * bar_width)


fig, ax = plt.subplots(figsize=(14,8))

# Plot the bars

plt.bar(x - 2 * bar_width , smefit_bounds_lo_nho, bar_width, align='center', label=r'$\rm{LO}, \mathcal{O}\left(\Lambda^{-2}\right)$')
plt.bar(x - bar_width, smefit_bounds_lo_ho, bar_width, align='center', label=r'$\rm{LO, \mathcal{O}\left(\Lambda^{-4}\right)}$')
plt.bar(x, smefit_bounds_nlo_nho, bar_width, align='center', label=r'$\rm{NLO}, \mathcal{O}\left(\Lambda^{-2}\right)$')
plt.bar(x + bar_width, smefit_bounds_nlo_ho, bar_width, align='center', label=r'$\rm{NLO, \mathcal{O}\left(\Lambda^{-4}\right)}$')
plt.bar(x + 2 * bar_width , fitmaker_bounds, bar_width, align='center', label=r'$\rm{Fitmaker}$')

# Set labels and title

plt.ylabel(r'$|g_{\mathrm{UV}}|$')

# Set x-axis tick labels
plt.xticks(x, categories)

# Add a legend
plt.legend(frameon=False, ncol=3)

# Show the plot
plt.savefig('uv_smefit_vs_fitmaker_all.pdf')
