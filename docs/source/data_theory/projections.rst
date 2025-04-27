Projections
===========
As of SMEFiT3.0, one can project existing measurements to other running scenarios with reduced statistical and systematic
uncertainties. This is relevant for projection studies for instance at HL-LHC or other future colliders such as the FCC-ee.
In the following, we first lay out the theory behind this, followed by how one can use it in SMEFiT.

Theory
------

The projection module starts by considering a given available measurement from the LHC Run  II, composed by :math:`n_{\rm bin}` data points, and with the corresponding theory predictions given by :math:`\mathcal{O}_i^{{\rm (th)}}`.
These can either be SM or BSM predictions.
The central values for the pseudo-data, denoted by :math:`\mathcal{O}_i^{{\rm (exp)}}`, are obtained
by fluctuating these theory predictions by the fractional statistical :math:`(\delta_i^{\rm (stat)})`
and systematic :math:`(\delta_{k,i}^{\rm (sys)})` uncertainties,

.. math::
    \begin{equation}
      \label{eq:pseudo_data_v2}
      \mathcal{O}_i^{{\rm (exp)}}
      = \mathcal{O}_i^{{\rm (th)}}
        \left( 1+ r_i \delta_i^{\rm (stat)}
        + \sum_{k=1}^{n_{\rm sys}}
        r_{k,i} \delta_{k,i}^{\rm (sys)}
        \right) \,
        , \qquad i=1,\ldots,n_{\rm bin} \, ,
     \end{equation}
where :math:`r_i` and :math:`r_{k,i}` are univariate random Gaussian numbers, whose distribution is such as to reproduce
the experimental covariance matrix of the data, and the index :math:`k` runs over the individual sources of correlated
systematic errors. We note that theory uncertainties are not included in the pseudo-data generation, and enter only
the calculation of the :math:`\chi^2`.

Since one is extrapolating from an existing measurement, whose associated statistical and systematic errors are denoted
by :math:`\tilde{\delta}_i^{\rm (stat)}` and :math:`\tilde{\delta}_{k,i}^{\rm (sys)}`, one needs to account for the
increased statistics and the expected reduction of the systematic uncertainties for the HL-LHC data-taking period.
The former follows from the increase in integrated luminosity,

.. math::
    \begin{equation}
        \delta_i^{\rm (stat)} = \tilde{\delta}_i^{\rm (stat)} \sqrt{\frac{\mathcal{L}_{\rm Run2}}{\mathcal{L}_{\rm HLLHC}}} \,, \qquad i=1,\ldots, n_{\rm bin} \, ,
    \end{equation}
while the reduction of systematic errors is estimated by means of an overall rescaling factor

.. math::
    \begin{equation}
        \delta_{k,i}^{\rm (sys)} = \tilde{\delta}_{k,i}^{\rm (sys)}\times f_{\rm red}^{(k)} \,, \qquad i=1,\ldots, n_{\rm bin} \, ,\quad k=1,\ldots, n_{\rm sys} \, .
    \end{equation}

with :math:`f_{\rm red}^{(k)}` indicating a correction estimating improvements in the experimental performance,
in many cases possible thanks to the larger available event sample. Here for simplicity we adopt the optimistic scenario
considered in the HL-LHC projection studies :cite:`Cepeda:2019klc`, namely :math:`f_{\rm red}^{(k)}=1/2` for all the datasets.


For datasets without the breakdown of statistical and systematic errors,
Eq.~(\ref{eq:pseudo_data_v2}) is replaced by

.. math::
    \begin{equation}
        \label{eq:pseudo_data_v3}
        \mathcal{O}_i^{{\rm (exp)}}
        = \mathcal{O}_i^{{\rm (th)}}
            \left( 1+ r_i \delta_i^{\rm (tot)}
            \right) \,
            , \qquad i=1,\ldots,n_{\rm bin} \, ,
    \end{equation}

with the total error being reduced by a factor :math:`\delta_i^{\rm (tot)}=f_{\rm red}^{{\rm tot}} \times \tilde{\delta}_i^{\rm (tot)}`
with :math:`f_{\rm red}^{{\rm tot}}\sim 1/3`, namely the average of the expected reduction of statistical and systematic
uncertainties as compared to the baseline Run II measurements. For such datasets, the correlations are neglected in the projections due to the lack of their  breakdown.

Creating projections
--------------------
Fits with projections follow a two-step process. First, one creates projected datasets (typically with reduced statistical
and systematic uncertainties) as ``.yaml`` files in the standard SMEFiT format with the following command,

.. code-block:: bash

    smefit PROJ --lumi <luminosity> --noise <noise level> /path/to/projection_runcard.yaml

where ``<luminosity>`` specifies the luminosity of the projection in :math:`{\rm fb}^{-1}`. The noise level ``<noise level>``
can be either ``L0`` are ``L1`` corresponding to either level 0 or level 1 projections respectively. In level 0 projections,
the experimental central value coincides exactly with the theory prediction, while the experimental central values are fluctuated around
the theory prediction according to the experimental uncertainties in case of level 1. If ``<noise level>`` is not specified, level 0
is assumed. If ``<luminosity>`` is not specified, the original luminosities are kept and the uncertainties are not rescaled.
The ``projection_runcard`` specifies which datasets need to be extrapolated, by which factor to reduce the systematics, and sets the necessary paths:

.. code-block:: yaml

        # path to where projections get saved
        projections_path: /path/to/projected_data

        # path to existing data
        commondata_path: /path/to/exisiting_data

        # path to theory tables
        theory_path: /path/to/theory_tables

        # datasets for which projections are computed
        datasets:
          - {name: ATLAS_tt_13TeV_ljets_2016_Mtt, order: NLO_QCD}
          - {name: CMS_ggF_aa_13TeV, order: NLO_QCD}
          - {name: LEP1_EWPOs_2006, order: LO}

        coefficients:
          OtG: {constrain: True, value: 2.0}
          OpD: {constrain: True, value: -1.0}

        uv_couplings: False
        use_quad: False
        use_theory_covmat: False
        rot_to_fit_basis: null
        use_t0: True # use the t0 prescription to correct for d'Agostini bias

        fred_sys: 0.5 # systematics get reduced by 1/2
        fred_tot: 0.333 # total errors get reduced by 1/3

If the coefficients are not specified, the predictions will be computed at the SM point.

The projected datafiles will get appended the suffix ``_proj`` so that they can be easily distinguished from the original
ones. The corresponding theory file (which is the same for both the projected and the original datasets) also gets appended
this same suffix.

Once the projected datasets are written at the specified ``projections_path``, one can use these in exactly the same way
as the original datasets. They can be read by SMEFiT directly.
