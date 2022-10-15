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
The sys_name can be CORR, UNCORR to specify whether the systematic considered id correlated or uncorrelated
within the dataset. In other words, if sys_name=CORR for sys1, then sys1 will be correlated across the datapoints
of the dataset. Same for UNCORR.

The same logic can be used for CORR and UNCORR theory systemtics within a dataset, using the names THEORYCORR and THEORYUNCORR.

For systematics correlated between different dataset a different name has to be used, which must be
the same for the corresponding systematic in all the datasets. 
