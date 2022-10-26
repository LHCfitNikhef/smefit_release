```eval_rst
.. _data:
```

# Experimental data format
Experimental data should be provided in `.yaml` file

```yaml
dataset_name: example_dataset
num_data: 3
num_sys: 2
data_central:
- data1
- data2
- data3
statistical_error:
- sys1
- sys2
- sys3
systematics:
- - sys1_data1
  - sys1_data2
  - sys1_data3
- - sys2_data1
  - sys2_data2
  - sys2_data3
sys_names:
- CORR
- CORR
sys_type:
- MULT
- ADD
```
The systematic name can be ``CORR``, ``UNCORR`` to specify whether the systematic considered is correlated or uncorrelated within the dataset.
In the same way ``THEORYCORR`` and ``THEORYUNCORR`` can be used for correlated and uncorrelated theory systematics within a dataset.

For uncertainties correlated between different dataset a different name has to be used, which must be
the same for the corresponding systematic in all the datasets. For the details about the construction of the 
covariance matrix from the list of statistic and systematic uncertainty see [here](./covariance.html#construction-of-the-covariance-matrix).

For some dataset only the full covariance matrix might be available. In order to use the dataset within the ``smefit`` code, the user has to decompose it in a set of correlated systematics.
In [this](./decomposition_cov.html) section we provide guidelines on how to do this.
