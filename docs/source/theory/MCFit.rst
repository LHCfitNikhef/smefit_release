The Monte Carlo replica method
==============================

Formalism
~~~~~~~~~

The Monte Carlo replica approach (MCfit), which in turn was inspired by the
NNPDF analysis of the quark and gluon substructure of protons.

This method aims to construct a sampling of the probability
distribution in the space of the experimental data, which then translates
into a sampling of the probability distribution in the space of the EFT
coefficients through an optimisation procedure where the best-fit values
of the coefficients for each replica, :math:`\boldsymbol{c}^{(k)}`, are determined.

Given an experimental measurement of a hard-scattering cross-section, denoted by :math:`\sigma_i^{\rm (exp)}`,
with total uncorrelated uncertainty :math:`\delta_{i}^{\rm (stat)}` and :math:`n_{\rm sys}`
correlated systematic uncertainties :math:`\delta^{\rm (sys)}_{i,\alpha}`, the :math:`N_{\rm rep}` artificial
Monte Carlo (MC) replicas of the experimental data are generated as

.. math::
    \sigma_{i}^{(\mathrm{art})(k)} = \sigma_{i}^{\rm (exp)}\left( 1 + r_{i}^{(k)}\delta_{i}^{\rm (stat)} +
    \sum_{\alpha=1}^{n_{\rm sys}}r_{i,\alpha}^{(k)}\delta^{\rm (sys)}_{i,\alpha}\right)
    \quad k=1,\ldots,N_{rep}

where the index :math:`i` runs from 1 to :math:`n_{\rm dat}` and :math:`r_{i}^{(k)}`, :math:`r_{i,\alpha}^{(k)}`
are univariate Gaussian random numbers.
Correlations between data points induced by systematic uncertainties
are accounted for by ensuring that :math:`r^{(k)}_{i,\alpha}=r^{(k)}_{i',\alpha}`.


It can be show that central values, variances, and covariances evaluated
by averaging over the MC replicas reproduce the corresponding experimental values.

A fit to the :math:`n_{\rm op}` degrees of freedom :math:`\boldsymbol{c}/\Lambda`
is then performed for each of the MC replicas :math:`\sigma_{i}^{(\mathrm{art})(k)}` generated.

The best-fit values are determined from the minimisation of the cost function

.. math::
  E^{(k)}({\boldsymbol c})\equiv \frac{1}{n_{\rm dat}}\sum_{i,j=1}^{n_{\rm dat}}\left(
  \sigma^{(\rm th)}_i\left( {\boldsymbol c}^{(k)}\right ) -\sigma^{{(\rm art)}(k)}_i\right ) ({\rm cov}^{-1})_{ij}
  \left ( \sigma^{(\rm th)}_j\left ( {\boldsymbol c}^{(k)} \right )-\sigma^{{(\rm art)}(k)}_j\right )

where :math:`\sigma^{(\rm th)}_i( {\boldsymbol c}^{(k)} )` indicates the theoretical
prediction for the `i`-th cross-section evaluated with the `k`-th set of EFT coefficients.

This process results in a collection of :math:`{\boldsymbol c}^{(k)}` best-fit
coefficient values from which estimators such as expectation values, variances,
and correlations are evaluated.

The overall fit quality is then evaluated using the :math:`\chi^2` definition,
where the central experimental values are compared to the mean theoretical
prediction computed by the resulting fit replicas.


Various theoretical uncertainties are also included in the :math:`\chi^2` definition for some datasets.
A consistent treatment of theoretical uncertainties in the fitting procedure means
that these are not only included in the fit via
the covariance matrix in :math:`\chi^2` definition, but also in the corresponding replica generation.
In other words, the replicas are sampled according to a multi-Gaussian distribution
defined by the total covariance matrix which receives contributions both of experimental and of theoretical origin.
We therefore account for such errors in the generation of Monte Carlo replicas :math:`\sigma_{i}^{(\mathrm{art})(k)}`,
:cite:`thennpdfcollaboration2019parton`.

There are numerous advantages of using the MCfit method for global EFT analyses:

    *   it does not require specific assumptions about the underlying probability distribution
        of the fit parameters, and in particular does not rely on the Gaussian approximation.
    *   the computational cost scales in a much milder way with the number of operators
        :math:`n_{\rm op}` included in the fit as compared to NS.
    *   Thirdly, it can be used to assess the impact of new datasets in the fit `a posteriori`
        with the Bayesian reweighting formalism.

However, it only works for linear EFT fits, as first explained in :cite:`Kassabov:2023hbm`.

Optimisation
~~~~~~~~~~~~

In the top quark sector analysis of :cite:`Hartland:2019bjb`, the minimisation of
Eq.~\eqref{eq:chi2definition} was achieved by a gradient descent method which relies
on local variations of the error function.
This choice is advantageous since :math:`E^{(k)}` is at most a quartic form
of the fit parameters, and therefore evaluating its gradient is computationally efficient.

In the current version of the code ( see :cite:`ethier2021combined` analysis) we allow for a more complex parameter space,
using as optimiser a trust-region algorithm ``trust-constr`` available in the ``SciPy`` package.
An advantage of this method is that it allows one to provide the optimiser with any combination of constraints on the
coefficients, including existing bounds.
This is a rather useful feature, since in many cases of interest one would like to restrict
the EFT parameter space based on theoretical considerations, such as when
accounting for the LEP EWPOs or in the top-philic scenario.



Postfit selection criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~

One disadvantage of optimisation strategies such as MCfit is that as the parameter space
space is increased, the minimiser might sometimes converge on a local,
rather than on the global, minimum.
This is specially problematic in the quadratic EFT fits which often display
quasi-degenerate minima.
For this reason, it is important to implement post-fit quality selection criteria
that indicate when a fitted replica should be kept and when it should be discarded.
Here, a MC replica is kept if the total error function of the replica dataset, :math:`E_{\rm tot}^{(k)}`,
satisfies

.. math::
    E_{\rm tot}^{(k)}\le 3
