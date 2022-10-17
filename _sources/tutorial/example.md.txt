```eval_rst
.. _example:
```
Tutorial
========

In the following we provide detailed instructions on how to use the code in its different 
running modes and on how to analyse the results.

# Runcard specifications
The basic object required to run the code is a runcard. 
In this section we document the different parameters which have to be specified here. 
As example we will refer to the runcard to reproduce ``smefit2.0``,
available from the repository [smefit_database](https://github.com/LHCfitNikhef/smefit_database)
together with the files containing experimental data and the corresponding theory predictions.
After cloning the repository, run 
```yaml
python update_runcards_path.py -d /path/to/runcard/destination/ runcards/NS_GLOBAL_NLO_NHO.yaml 
```
This will create in ``/path/to/runcard/destination/`` a ``smefit2.0`` runcard ready to be used on the local machine of the user, pointing to the experimental data and an theory tables in the repository smefit_database.
In the folder ``smefit_database/runcards`` the input runcards for MC and NS fits with both linear (NHO)
and linear+quadratic corrections (HO) are available.


## Input and output path
The path to where the experimental data and the corresponding theory tables are stored
is automatically set to those contained in ``smefit_dayabase`` by the script ``update_runcards_path.py``.
The user can change them manually if other set of data are desired. 
The folder where the results will be saved can be set using ``result_path``. The file containing 
the posterior of the fitted Wilson coefficient will be saved in ``resulth_path/result_ID``. 
If ``result_ID`` is not provided, it will be automatically set to the name of the runcard (and any already existing result will be overwritten).

```yaml
result_ID: 
result_path:
data_path:  
theory_path: 


```
## Theory specifications
The perturbative order of the QCD theory prediction (LO or NLO) should be specified using ``order``.
``use_quad`` should be set to ``True`` for a fit with quadratic corrections, ``use_t0`` controls the use 
of the ``t0`` prescription and ``use_theory_covmat`` specifies whether or not to use the theory covariance matrix
which can be specified in the theory files. 

```yaml
order: NLO
use_quad: False
use_t0: False
use_theory_covmat: True
```

## Minimizer specifications
The different parameters controlling the minimizer used in the analysis are specified here.
If ``single_parameter_fits`` is set to ``True``, the Wilson coefficient specified in the runcard
will be fit one at time, setting all the other to 0. See [here](./example.html#single-parameter-fits) for more details

```yaml
single_parameter_fits: False
bounds: Null

# NS settings
nlive: 1000
efr: 0.005
ceff: True
toll: 0.5

#MC settings
mc_minimiser: 'cma'
restarts: 7
maxiter: 100000
chi2_threshold: 3.0
```

## Datasets to consider and coefficients to fit
The datasets and Wilson coefficients to be included in the analysis must be listed under ``datasets``
and ``coefficients`` respectively.

```yaml
datasets:

  - ATLAS_tt_8TeV_ljets_Mtt
  - ATLAS_tt_8TeV_dilep_Mtt
  - CMS_tt_8TeV_ljets_Ytt
  - CMS_tt2D_8TeV_dilep_MttYtt
  - CMS_tt_13TeV_ljets_2015_Mtt
  - CMS_tt_13TeV_dilep_2015_Mtt
  - CMS_tt_13TeV_ljets_2016_Mtt
  - CMS_tt_13TeV_dilep_2016_Mtt
  - ATLAS_tt_13TeV_ljets_2016_Mtt
  - ATLAS_CMS_tt_AC_8TeV
  - ATLAS_tt_AC_13TeV
  ... 
  ...

# Coefficients to fit
coefficients:

  OpQM: {'min': -10, 'max': 10}
  O3pQ3: {'min': -2.0, 'max': 2.0}
  Opt: {'min': -25.0, 'max': 15.0}
  OtW: {'min': -1.0, 'max': 1.0}
  ...
  ...
  
```

As exemplified above, the syntax to specify the Wilson coefficient corresponding to the operator 
``O1`` is ``O1 : {'min': , 'max':} `` where ``min`` and ``max`` indicate the bounds within which NS
will perform the sampling.  

## Constrains between coefficients
Some Wilson coefficients are not directly fit, but rather constrained to be linear combinations
of the other ones. Taking as example some coefficient of the Higgs sector considered in ``smefit2.0``, 
this can be specified in the runcard in the following way 

```yaml
  OpWB: {'min': -0.3, 'max': 0.5}
  OpD: {'min': -1.0, 'max': 1.0}
  OpqMi: {'constrain': [{'OpD': 0.9248},{'OpWB': 1.8347}], 'min': -30, 'max': 30}
  O3pq: {'constrain': [{'OpD': -0.8415},{'OpWB': -1.8347}], 'min': -0.5, 'max': 1.0}
```
In this example the coefficients ``CpqMi`` and ``C3pq`` correspondong to the operators
``OpqMi`` and ``O3pq`` will be set to
```yaml
CpqMi = 0.9248*CpD + 1.8347*CpWB 
C3pq = -0.8415*CpD -1.8347*CpWB 
```

## Fit in different basis
It is possible to run a fit using a different basis. 
In this case the coefficients specified in the runcard should be the ones to fit, and 
a rotation basis defining the new basis in terms of the Warsaw basis should be given as input. This can be done
using the option below.

```yaml
rot_to_fit_basis: /path/to/rotation/rotation.json

```


# Running a fit with NS
To run a fiy using Nested Sampling use the command
```yaml
smefit NS path/to/the/runcard/runcard.yaml
```

This will generate a file named ``posterior.json`` in the result folder, 
containing the posterior distribution of the coefficients specified in the runcard.

# Running a fit with MC
The basic command to run a fit using Monte Carlo is  
```yaml 
smefit MC path/to/the/runcard/runcard.yaml -n <replica_number>
```

This will produce a file called ``replica_<replica_number>/coefficients_rep_<replica_number>.json`` 
in the result folder, containing the values of the Wilson coefficients for the replica. 
Once an high enough number of replicas have been produced, the results can be merged into the final posterior
running PostFit
```yaml
smefit PF path/to/the/result/ -n <number_of_replicas>
```
where ``<number_of_replicas>`` specifies the number of replicas to be used to build the posterior. 
Replicas not satisfying the PostFit criteria will be discarded. If the final number of good replicas is lower than 
``<number_of_replicas>`` the code will output an error message asking to produce more replicas first.
The final output is the file ``posterior.json`` containing the full posterior of the Wilson coefficients. 

# Single parameter fits
Given a runcard with a number of Wilson coefficients specified, it is possible to fit each of them in turn,
keeping all the other ones fix to 0. 
To do this add to the runcard 
```yaml
single_parameter_fits: True
``` 
and proceed as documented above for a normal fit. 
For both NS and MC, the final output will be the file ``posterior.json``
containing the independent posterior of the fitted Wilson coefficients, obtained by a series os independent single parameter fits.


# Producing a report
Once the file containing the posterior has been produced, the results can be visualized by running a report,
which can be used for both analysing a single fit or comparing results from multiple ones.
The report details have to be specified in a separate runcard, and the report is produced by running
```yaml
smefit R /path/to/report/runcard/runcard.yaml
```
The runcard used to analyse the ``smefit2.0`` results can be obtained from the ``smefit_database`` repository,
and it is take here as an example. 

The report will be saved in ``report_path/name`` and will compare the fits having ``result_ID``
``NS_GLOBAL_NLO_NHO`` and ``MS_GLOBAL_NLO_NHO``, whose results are therefore saved in ``result_path/NS_GLOBAL_NLO_NHO`` and ``result_path/MS_GLOBAL_NLO_NHO``
respectively


```yaml
name: "smefit2.0_NS_vs_MC_linear"

title: "Comparison between linear analysis using NS and MC"

result_IDs: [
  "NS_GLOBAL_NLO_NHO",
  "MC_GLOBAL_NLO_NHO",
]

fit_labels: [
  'smefit2.0 NS',
  'smefit2.0 MC',
]

report_path: 
result_path: 
```

```yaml
summary: True
```

```yaml
coefficients_plots:

  scatter_plot:
    figsize: [10,15]
    x_min: -50
    x_max: 50
    lin_thr: .01
    x_log: True
        
  confidence_level_bar:
    confidence_level: 95
    figsize: [10,15]
    plot_cutoff: 400
    x_min: 0.001
    x_max: 500
    x_log: True

  contours_2d:
    show: True
    confidence_level: 95
    dofs_show: ["O3pQ3", "OpQM"] # Null or list of op per fit to be displayed

  posterior_histograms: True

  table: True

  logo: True
  show_only: Null
  hide_dofs: Null # Null or list

  double_solution:  # List of op per fit with double solution
    NS_GLOBAL_NLO_NHO: []
    MC_GLOBAL_NLO_NHO: []
```

```yaml
correlations:
   hide_dofs: Null # Null or list of op not be displayed
   thr_show: 0.1 # Min value to show, if Null show the full matrix
```
```yaml
PCA:
  table: True # display the list of PC decomposition
  thr_show: 1.e-2
  sv_min: 1.e-4
  plot: True
  fit_list: [NS_GLOBAL_NLO_NHO] 
```

```yaml
chi2_plots:
  table: True # bool, chi2 table
  plot_experiment:  # chi2 plot per experiment
    figsize: [10,15]
  plot_distribution: #  chi2 distribution per replica
    figsize: [7,5]

```
The latex names of the Wilson coefficients entering the analysis
should be specified under ``coeff_info`` using the following syntax 
```yaml
coeff_info:
  4H: [
    [OQQ1, "$c_{QQ}^{1}$"],
    [OQQ8, "$c_{QQ}^{8}$"],
    [OQt1, "$c_{Qt}^{1}$"],
    [OQt8, "$c_{Qt}^{8}$"],
    [OQb1, "$c_{Qb}^{1}$"],
    [OQb8, "$c_{Qb}^{8}$"],
    [Ott1, "$c_{tt}^{1}$"],
    [Otb1, "$c_{tb}^{1}$"],
    [Otb8, "$c_{tb}^{8}$"],
    [OQtQb1, "$c_{QtQb}^{1}$"],
    [OQtQb8, "$c_{QtQb}^{8}$"],
  ]
  2L2H: [
    [O81qq, "$c_{qq}^{1,8}$"],
    [O11qq, "$c_{qq}^{1,1}$"],
    [O83qq, "$c_{qq}^{8,3}$"],
    [O13qq, "$c_{qq}^{1,3}$"],
    [O8qt, "$c_{qt}^{8}$"],
    [O1qt, "$c_{qt}^{1}$"],
    [O8ut, "$c_{ut}^{8}$"],
    [O1ut, "$c_{ut}^{1}$"],
    [O8qu, "$c_{qu}^{8}$"],
    [O1qu, "$c_{qu}^{1}$"],
    [O8dt, "$c_{dt}^{8}$"],
    [O1dt, "$c_{dt}^{1}$"],
    [O8qd, "$c_{qd}^{8}$"],
    [O1qd, "$c_{qd}^{1}$"],
  ]
  ...
  ...
```
Similarly the information corresponding to the experimental data entering the analysis 
should be reported as in the following
```yaml
data_info:
  tt8: [
    [ATLAS_tt_8TeV_ljets_Mtt, https://arxiv.org/abs/1511.04716],
    [ATLAS_tt_8TeV_dilep_Mtt, https://arxiv.org/abs/1607.07281],
    [CMS_tt_8TeV_ljets_Ytt, https://arxiv.org/abs/1505.04480],
    [CMS_tt2D_8TeV_dilep_MttYtt, https://arxiv.org/abs/1703.01630],
  ]
  tt13: [
    [CMS_tt_13TeV_ljets_2015_Mtt, https://arxiv.org/abs/1610.04191],
    [CMS_tt_13TeV_dilep_2015_Mtt, https://arxiv.org/abs/1708.07638],
    [CMS_tt_13TeV_ljets_2016_Mtt, https://arxiv.org/abs/1803.08856],
    [CMS_tt_13TeV_dilep_2016_Mtt, https://arxiv.org/abs/1811.06625],
    [ATLAS_tt_13TeV_ljets_2016_Mtt, https://arxiv.org/abs/1908.07305],
  ]
...
...
```
The names of both the operators and the datasets are those used in the fit runcard.


# SCAN
The code can also be used to produce 1-dimensional scans of the chi2 function.
The command

```yaml
smefit SCAN /path/to/the/runcard/runcard.yaml
```
will produce in the results folder a series of pdf files containing plots for
1-dimensional scans of the chi2 with respect to each parameter in the runcard.
