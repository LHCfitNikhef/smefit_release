Projections
===========
As of SMEFiT3.0, one can project existing measurements to other running scenarios with reduced statistical and systematic
uncertainties. This is relevant for projection studies for instance at HL-LHC or other future colliders such as the FCC-ee.
In the following, we first lay out the theory behind this, followed by how one can use it in SMEFiT.

Theory
------

The projection module starts by considering a given available measurement from the LHC Run  II, composed by :math:`n_{\rm bin}` data points, and with the corresponding SM predictions given by :math:`\mathcal{O}_i^{{\rm (th)}}`.
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

    smefit proj --lumi <luminosity> /path/to/projection_runcard.yaml

where the desired luminosity in :math:`{\rm fb}^{-1}` can be specified by replacing ``<luminosity>``. The ``projection_runcard``
specifies which datasets need to be extrapolated, by which factor to reduce the systematics, and sets the necessary paths:

.. code-block:: yaml

        # path to where projections get saved
        projections_path: /path/to/projected_data

        # path to existing data
        commondata_path: /path/to/exisiting_data

        # path to theory tables
        theory_path:  /path/to/theory_tables

        # datasets for which projections are computed
        datasets:

          # TOP QUARK PRODUCTION
          # ttbar
          - ATLAS_tt_13TeV_ljets_2016_Mtt
          - CMS_tt_13TeV_dilep_2016_Mtt
          - CMS_tt_13TeV_Mtt
          - CMS_tt_13TeV_ljets_inc

          # ttbar asymm and helicity frac
          - ATLAS_tt_13TeV_asy_2022_uncor
          - CMS_tt_13TeV_asy
          - ATLAS_Whel_13TeV_uncor

          # ttbb
          - ATLAS_ttbb_13TeV_2016
          - CMS_ttbb_13TeV_2016
          - CMS_ttbb_13TeV_dilepton_inc
          - CMS_ttbb_13TeV_ljets_inc

          # tttt
          - ATLAS_tttt_13TeV_run2
          - CMS_tttt_13TeV_run2
          - ATLAS_tttt_13TeV_slep_inc
          - CMS_tttt_13TeV_slep_inc
          - ATLAS_tttt_13TeV_2023
          - CMS_tttt_13TeV_2023

          # ttZ
          - CMS_ttZ_13TeV_pTZ
          - ATLAS_ttZ_13TeV_pTZ_uncor

          # ttW
          - ATLAS_ttW_13TeV_2016
          - CMS_ttW_13TeV

          # Single top
          - ATLAS_t_tch_13TeV_inc
          - CMS_t_tch_13TeV_2019_diff_Yt
          - ATLAS_t_sch_13TeV_inc

          # tW
          - ATLAS_tW_13TeV_inc
          - CMS_tW_13TeV_inc
          - CMS_tW_13TeV_slep_inc

          # tZ
          - ATLAS_tZ_13TeV_run2_inc
          - CMS_tZ_13TeV_pTt_uncor

          # HIGGS PRODUCTION

          # Signal Strengths
          - ATLAS_SSinc_RunII
          - CMS_SSinc_RunII

          # ATLAS & CMS Run II Higgs Differential
          - CMS_H_13TeV_2015_pTH

          # ATLAS & CMS STXS
          - ATLAS_WH_Hbb_13TeV
          - ATLAS_ZH_Hbb_13TeV
          - ATLAS_ggF_13TeV_2015
          - ATLAS_ggF_ZZ_13TeV
          - CMS_ggF_aa_13TeV
          - ATLAS_STXS_runII_13TeV_uncor

          # DIBOSON DATA
          - ATLAS_WW_13TeV_2016_memu
          - ATLAS_WZ_13TeV_2016_mTWZ
          - CMS_WZ_13TeV_2016_pTZ
          - CMS_WZ_13TeV_2022_pTZ


        order: NLO
        use_quad: False
        use_theory_covmat: False
        rot_to_fit_basis: null
        use_t0: True # use the t0 prescription to correct for d'Agostini bias

        fred_sys: 0.5 # systematics get reduced by 1/2
        fred_tot: 0.333 # total errors get reduced by 1/3

The projected datafiles will get appended the suffix ``_proj`` so that they can be easily distinguished from the original
ones. The corresponding theory file (which is the same for both the projected and the original datasets) also gets appended
this same suffix.

Once the projected datasets are written at the specified ``projections_path``, one can use these in exactly the same way
as the original datasets. They can be read by SMEFiT directly.

In case the original luminosity needs to be kept and one is only interested in adding statistical noise, one should use the following
syntax

.. code-block:: bash

    smefit proj --closure /path/to/projection_runcard.yaml

This does nothing to the statistical and systematic uncertainties - it only fluctuates the central value around the SM
prediction according to the specified uncertainties.
