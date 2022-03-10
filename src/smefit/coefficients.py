# -*- coding: utf-8 -*-
from collections import namedtuple

import numpy as np

# TODO consider using a DataClass and always read coefficients properties
# from this class and not from the config

CoeffTuple = namedtuple("Coefficients", ("labels", "values", "bounds", "fixed"))


def aggregate_coefficients(input_coefficients, loaded_datasets, input_bounds=None):
    """
    Aggregate all coefficient labels and construct an array of coefficient
    values of suitable size. Returns a CoeffTuple of the labels, values,
    and bounds

    Parameters
    ----------
        config : dict
            config dictionary
        loaded_datasets : DataTuple
            loaded datasets
    Returns
    -------
        CT_CoeffTuple : CoeffTuple
            CoeffTuple of the labels, values and bounds
    """

    # Give the initial point of the fit to be randomly spread around the bounds
    # specified by --bounds option (if none given, bounds are taken from setup.py)
    dataset_coefficients = []

    # for set in loaded_datasets:
    for key in loaded_datasets.CorrectionsKEYS:
        dataset_coefficients.append(key)

    # Keep ordering of coefficients the same so they match to the actual corrections
    dataset_coefficients, idx = np.unique(
        np.array(dataset_coefficients), return_index=True
    )
    dataset_coefficients = dataset_coefficients[np.argsort(idx)]

    # All the coefficients are initialized to 0 by default
    values = np.zeros(len(dataset_coefficients))
    bounds = [(0.0, 0.0) for i in range(0, len(dataset_coefficients))]
    fixed = [True for i in range(0, len(dataset_coefficients))]

    # for k in config["coefficients"].keys():
    for k in input_coefficients.keys():
        if k not in dataset_coefficients:
            raise ValueError(
                f"{k} is not part of fitted coefficients. Please comment it out in the setup file"
            )

        if input_bounds is None:
            min_val = input_coefficients[k]["min"]
            max_val = input_coefficients[k]["max"]

        idx = np.where(dataset_coefficients == k)[0][0]
        bounds[idx] = (min_val, max_val)
        values[idx] = np.random.uniform(low=min_val, high=max_val)
        fixed[idx] = input_coefficients[k]["fixed"]

    return CoeffTuple(dataset_coefficients, values, bounds, fixed)
