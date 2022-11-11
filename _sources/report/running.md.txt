```eval_rst
.. _running_report:
```

# Produce a report

Once the file containing the posterior has been produced, the results can be visualized by running a report,
which can be used for both analysing a single fit or comparing results from multiple ones.
The report details have to be specified in a separate runcard, and the report is produced by running

```bash
  smefit R /path/to/report/runcard.yaml
```
All the plots and tables will be stored in a folder and are avialabe both in HTML and pdf.

## Runcard descriprion

Here we describe the various runcard entries that can be set to produce a report.
The runcard is a `.yaml` file where in the first section we specify the input fits we are trying to compare.
All the following entries are **required**.

```yaml
    # report folder name
    name: "report_folder_name"

    # report title
    title: "Comparison between fit 1 and fit 2"

    # fit namen
    result_IDs: [
      "fit_1",
      "fit_2",
    ]

    # fit labels displayed in the plots
    fit_labels: [
      "smefit fit 1",
      "smefit fit 2",
    ]

    # path where the report will be saved
    report_path: path/to/report/
    # path where the reults are located
    result_path: path/to/results/
```

The report will be saved in ``report_path/name`` and will compare the fits having
``result_ID=fit_1`` and ``result_ID=fit_2``,
whose results are therefore saved in ``result_path/fit_1`` and ``result_path/fit_2`` respectively.

In the second part of the report runcard you can specify which kind of plot you want to
include in your report and with which settings.
All the  entries without a value are **optional**, if not present the correspoding analysis/plot will
be skipped.

```yaml
# include sumary tables of ffited coefficient and datasets
summary: True

# cofficient plots options
coefficients_plots:

  # scatter plot with central values and error bands
  scatter_plot:
    figsize: [10,15] # figure size
    x_min: -400 # x min value
    x_max: 400 # x min value
    lin_thr: .01 # linear threshold, in ca of x_log
    x_log: True # use symlog scale on x axis ?

  # confidence level error band plot
  confidence_level_bar:
    confidence_level: 95 # Avalable CL are 95% or 68%
    figsize: [10,15] # figure size
    plot_cutoff: 400 # value at which show a cut-off dashed line, defult is Null
    x_min: 0.01 # x min value
    x_max: 500 # x max value
    x_log: True # use log scale on x axis ?

  # 2 dimensional contour plot
  contours_2d:
    confidence_level: 95 # Avalable CL are 95% or 68%
    dofs_show: ["Op1", "Op2"] # list of operator to be displayed (will include all the possible pairs), default is Null

  # show the posterior histograms
  posterior_histograms: True

  # show a summary table with all the given bounds
  table:
    round_val: 3 # round values up to

  # display the smefit logo
  logo: True

  show_only: Null # list of operator to be displayed, default is all (Null)
  hide_dofs: Null # list of operator not to displayed, default is Null

  # emptry or list of operator per fit which have degenerated solution
  # (to be used only in qudtatic fits)
  double_solution:
    fit_1: ["Opdouble1", "Opdouble2"]
    fit_2: []

# correlation plot options
correlations:

  hide_dofs: Null # list of operator not to displayed, default is Null
  thr_show: 0.1 # Min value to show, if Null show the full matrix

# PCA options
PCA:

  fit_list: ["fit_1"] # list of fit for which PCA is performed, by default all the fits will be included
  table: True # display a table with the list of PC decomposition
  thr_show: 1.e-2 # min value of the principal component to display

  # heatmap plot
  plot:
    figsize: [15,15] # figure size
    sv_min: 1.e-4 # min singular value to display (upper plot)
    sv_max: 1.e+5 # max singular value to display (upper plot)
    thr_show: 0.1 # min value of the principal component to display (main plot)

# chi2 analysis plots and tables options
chi2_plots:

  table: True # display chi2 tables per dataset?

  # chi2 plot per dataset bar plot
  plot_experiment:
    figsize: [10,15] # figure size

  # chi2 distribution per replica histogram
  plot_distribution:
    figsize: [7,5] # figure size

# fisher information options
fisher:

  norm: "coeff" # normalize per "coeff" or "data"
  summary_only: True # if True display only the fisher information per dataset group. If False will show the fine grained dataset per dataset
  log: False # show values in log scale ?
  fit_list: ["fit_1"] # list of fit for which fisher is compued, by default all the fits will be included

  # heatmap plot
  plot:
    summary_only: True # if True display only the fisher information per dataset group. If False will show the fine grained dataset per dataset
    figsize: [11, 15] # figure size

```

Finally the user has to specify two dictionaries where the informaions about
Wilson coefficients and datasets entering the analysis are reporte are reported.
Both names of the operators and the datasets are those used in the fit runcard.
These informations are **required**.

Wilson coefficients latex names should be added, by group type following
the syntax:

```yaml
coeff_info:
  type1: [
    [Op1, "$c_{\\varphi}^{(1)}$"],
    [Op2, "$c_{\\varphi}^{(2)}$"],
    ...
  ]
  4H: [
    [OQQ1, "$c_{QQ}^{1}$"],
    [OQQ8, "$c_{QQ}^{8}$"],
    ...
  ]
  2L2H: [
    [O81qq, "$c_{qq}^{1,8}$"],
    [O11qq, "$c_{qq}^{1,1}$"],
    ...
  ]
  ...
```

Similarly the information corresponding to the experimental data
should be reported as in the following:

```yaml
data_info:
  group_1: [
    [dataset_1, https://lin/to/public/reference/1],
    [dataset_2, https://lin/to/public/reference/2],
    ...
  ]
  tt13: [
    [CMS_tt_13TeV_ljets_2015_Mtt, https://arxiv.org/abs/1610.04191],
    [CMS_tt_13TeV_dilep_2015_Mtt, https://arxiv.org/abs/1708.07638],
    [CMS_tt_13TeV_ljets_2016_Mtt, https://arxiv.org/abs/1803.08856],
    ...
  ]
  ...
```

You can see the smefit databse repo for futher examples about report runcards.
