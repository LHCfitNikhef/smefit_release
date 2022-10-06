Standard Model Effective Field Theory
=====================================

Here we present a brief description of the theoretical formalism
underlying the |SMEFT| framework :cite:`Weinberg:1978kz`  :cite:`Buchmuller:1985jz` and its application
to the analysis of particle physics data, see also :cite:`Brivio:2017vri` for a review.


The SMEFT framework
~~~~~~~~~~~~~~~~~~~

The effects of new heavy BSM particles with
typical mass scale :math:`M\simeq \Lambda` can under general conditions be
parametrized at lower energies :math:`E\ll \Lambda` in a model-independent way in
terms of a basis of higher-dimensional operators constructed from the SM fields
and their symmetries.
The resulting effective Lagrangian then admits the following power expansion

.. math::

   \mathcal{L}_{\rm SMEFT}=\mathcal{L}_{\rm SM} + \sum_i^{N_{d6}} \frac{c_i}{\Lambda^2}\mathcal{O}_i^{(6)} + \sum_j^{N_{d8}} \frac{b_j}{\Lambda^4}\mathcal{O}_j^{(8)} + \ldots \, ,


where :math:`\mathcal{L}_{\rm SM}` is the SM Lagrangian, and
:math:`\mathcal{O}_i^{(6)}` and :math:`\mathcal{O}_j^{(8)}` stand for
the elements of the
operator basis of mass-dimension d=6 and d=8,
respectively. Operators with d=5 and d=7, which violate lepton and/or baryon number
conservation :cite:`Degrande:2012wf`, are not considered here.
Whilst the choice of operator basis used in this expression is
not unique, it is possible to relate the results obtained in different
bases :cite:`Falkowski:2015wza`.
In our approach we adopt the Warsaw basis for :math:`\mathcal{O}_i^{(6)}`  :cite:`Grzadkowski:2010es`, and neglect effects arising from operators with mass dimension :math:`d\ge 8` .

For specific UV completions, the Wilson coefficients :math:`_i` in
can be evaluated in terms of the parameters of
the BSM theory, such as its coupling constants and masses.
However, in a
bottom-up approach, they are  a priori free parameters and they need to be
constrained from experimental data.
In general, the effects of the dimension-6 SMEFT operators in a given
observable, such as cross-sections at the LHC, differential distributions,
or other pseudo-observables, can be written as follows:

.. math::

   \sigma=\sigma_{\rm SM} + \sum_i^{N_{d6}}\kappa_i \frac{c_i}{\Lambda^2} + \sum_{i,j}^{N_{d6}}  \widetilde{\kappa}_{ij} \frac{c_ic_j}{\Lambda^4}  \, ,


where :math:`\sigma_{\rm SM}` indicates
the SM prediction and the Wilson coefficients :math:`c_i` are considered to be real
for simplicity.

In this equation, the second term arises from operators
interfering with the SM amplitude.
The resulting :math:`\mathcal{O}\left(\Lambda^{-2}\right)` corrections to the SM
cross-sections represent formally the dominant correction, though in many cases
they can be subleading for different reasons.
The third term in
representing :math:`\mathcal{O}\left(\Lambda^{-4}\right)` effects, arises from the
squared amplitudes of the |SMEFT| operators, irrespectively of whether or not the
dimension-6 operators interfere with the SM diagrams.
In principle, this
second term may not need to be included, depending on if the truncation at
:math:`\mathcal{O}\left(\Lambda^{-2}\right)` order is done at the Lagrangian or the cross
section level, but in practice there are often
valid and important reasons to include them in the calculation.


An important aspect of any |SMEFT| analysis is the need to include
all relevant operators that contribute
to the processes whose data is used as input
to the fit.
Only in this way can the |SMEFT| retain its
model and basis independence.
However, unless specific scenarios are adopted, the number of
non-redundant operators :math:`N_{d6}` becomes unfeasibly large:
59 for one generation of fermions :cite:`Grzadkowski:2010es` and 2499 for
three :cite:`Alonso:2013hga`.
This implies that a global |SMEFT| fit, even if
restricted to dimension-6 operators, will have to explore a huge
parameter space with potentially a large number of flat (degenerate) directions.
