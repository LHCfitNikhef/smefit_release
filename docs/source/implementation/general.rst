Fitting strategies
==================


In this section we describe the fitting methodology that is used to map the EFT parameter space.
|SMEFiT| implements one fitting strategies using the MultiNest and :doc:`Nested Sampling<NS>`
algorithm :cite:`Feroz:2007kg,Feroz:2013hea`, a robust sampling procedure is based on Bayesian inference.

In the following paragraphs we describe the log-likelihood function and the treatment of uncertainties, which are common
for the two fitting methodology. We also discuss the case of individual fits where just one EFT operator is take into account.

:math:`\chi^2` definition
-------------------------

The overall fit quality is quantified by the log-likelihood, or :math:`\chi^2` function, defined as

.. math::
  \chi^2 \left ( {\textbf{c}} \right ) \equiv \frac{1}{n_{\rm dat}}\sum_{i,j=1}^{n_{\rm dat}} \left (\sigma^{(\rm th)}_i \left( {\textbf{c}} \right) -\sigma^{(\rm exp)}_i\right) ({\rm cov}^{-1})_{ij}
  \left ( \sigma^{(\rm th)}_j \left( {\textbf{c}}\right) -\sigma^{(\rm exp)}_j\right)


where :math:`\sigma_i^{\rm (exp)}` and :math:`\sigma^{\rm (th)}_i \left(\textbf{c} \right )` are the central experimental data
and corresponding theoretical predictions for the `i-th` cross-section, respectively.


Uncertainties treatment
-----------------------

The total covariance matrix, :math:`{\rm cov}_{ij}`, should contain all relevant sources of
experimental and theoretical uncertainties. Assuming the latter are normally distributed, and that they are uncorrelated
with the experimental uncertainties, this total covariance matrix can be expressed as
a sum of the separate experimental and theoretical covariance


.. math::
    {\rm cov}_{ij} = {\rm cov}^{(\rm exp)}_{ij} + {\rm cov}^{(\rm th)}_{ij}

As usual, the experimental covariance matrix is constructed from all sources of statistical and systematic uncertainties that are
made available by the experiments. Moreover, the correlated multiplicative uncertainties are treated
via the `t_0` prescription :cite:`Ball:2009qv` in the fit, while the standard experimental definition is used to quote
the resulting :math:`\chi^2` values.

Concerning the theoretical covariance matrix, :math:`{\rm cov}^{(\rm th)}`, its contributions depend
on the specific type of processes considered.


Individual fits
---------------

Individual (one-parameter) fits correspond to varying a single EFT coefficient while keeping
the rest fixed to their SM values.
While such fits neglect the correlations between the different coefficients, they provide a useful
baseline for the global analysis, since there the CL intervals will be by construction looser (or at best, similar)
as compared to those of the one-parameters fits.

They are also computationally inexpensive, as they can be carried out analytically from a scan of the :math:`\chi^2`
profile without resorting to numerical methods.

Another benefit is that they facilitate the comparison between different
EFT analyses, which may adopt different fitting bases but whose individual bounds
should be similar provided they are based on comparable data sets and theoretical calculations.

In the scenario where a single EFT coefficient, :math:`c_j`, is allowed to vary while the rest are set to zero,
the theoretical cross-section (for :math:`\Lambda=1` TeV) given by simplifies to

.. math::
    \sigma_m^{\rm (th)}(c_j)= \sigma_m^{\rm (sm)} + c_j\sigma^{(\rm eft)}_{m,j} + c_j^2 \sigma^{(\rm eft)}_{m,jj}


which results in a quartic polynomial form for the :math:`\chi^2` when inserted into the :math:`\chi2` definition, namely:

.. math::
    \chi^2(c_j) = \sum_{k=0}^4 a_k \left(c_j\right)^k

Restricting the analysis to the linear order in the EFT expansion further simplifies to a parabolic form:

.. math::
    \chi^2(c_j) = \sum_{k=0}^2 a_k \left(c_j\right)^k = \chi^2_0 + b\left( c_j-c_{j,0} \right)^2

where :math:`c_{j,0}` is the value of :math:`c_j` at the minimum of the parabola,
and in this case linear error propagation (Gaussian statistics) is applicable.

To determine the values of the quartic polynomial coefficients :math:`a_k`, it is sufficient to fit this functional
form to a scan of the :math:`\chi^2` profile obtained by varying the EFT coefficient
:math:`c_j` when all other coefficients are set to their SM value.

The associated 95 % CL interval to the coefficient :math:`c_j` can then be determined by imposing the condition

.. math::
    \chi^2(c_j)-\chi^2(c_{j,0}) \equiv \Delta \chi^2 \le 5.91

We note that if the size of the quadratic :math:`\mathcal{O}\left(\Lambda^{-4}\right)` corrections is sizable,
there will be more than one solution for :math:`c_{j,0}` and one might end up with pairwise disjoint CL intervals.
