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
