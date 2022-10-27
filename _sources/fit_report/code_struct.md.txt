```eval_rst
.. _codestruct:
```

# Code structure

In the following the provide a quick overview of how the code works.
We refer to the [tutorial](./../tutorial/example.html) for a step by step guide on how to use it.

* The information required to set up the analysis are read from an input 
[runcard](./../tutorial/example.html#runcard-specifications).
The code will load the specified datasets, collecting the full experimental information and the 
Standard Model and EFT corrections to build the corresponding theory predictions.
In the same input card the operators whose Wilson coefficients should be fitted
are specified. If some of them do not appear in any of the selected datasets, the code will stop with an error message.

* Using the loaded information the loglikelihood is build as a function of the Wilson coefficients to fit.
This is then given as input to an ``Optimizer`` object, encoding all the specifications of the 
minimizer used in the analysis. 
The provided options are [Nested Sampling](./../tutorial/example.html#running-a-fit-with-ns) and 
[MonteCarlo](./../tutorial/example.html#running-a-fit-with-mc), based 
on the external MultiNest library and on an internal implementation respectively.
In the first case the output is the full posterior distribution for the fitted Wilson coefficients, 
while in the second is the result for a single MC replica. 
To get the full posterior using MC, after producing a sufficient number of replicas 
the code must be run in the PostFit mode, which discards replicas having ``chi^2`` 
higher than a certain threshold set by the user and produce the final output file.

* The final output of the fit is a json file having the following structure 
  ```yaml
  {
    O_1 : [sample_1, sample_2, ...],
    O_2 : [sample_1, sample_2, ...], 
    ...
  }
  ```
  where each entry correspond to a list of samples of the Wilson coefficient corresponding to the specified operator.
  The final posterior is therefore provided as a discrete set of samples for each fitted Wilson coefficient.

* The code can be run in the [Report mode](./../tutorial/example.html#producing-a-report) in order to produce a report to visualize the results. The different functionalities currently supported in the report are documented [here](./report_func.html)


