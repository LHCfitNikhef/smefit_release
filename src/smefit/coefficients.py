# -*- coding: utf-8 -*-

import numpy as np


class Coefficient:
    """
    Coefficient object

    Parameters
    ----------
        name: str
            name of the operator corresponding to the Wilson coefficient
        minimum : float
            mimimum value
        maximum : float
            maximum value
        value : float, optional
            best value. If None set to random between minimum and maximum
        constrain : dict, float, optional
            TODO: fix bound values
    """

    def __init__(self, name, minimum, maximum, value=None, constrain=None):
        self.op_name = name
        self.min = minimum
        self.max = maximum
        if value is None:
            self.value = np.random.uniform(low=minimum, high=maximum)
        self.constrain = constrain

    def __repr__(self):
        return self.op_name

    def __eq__(self, coeff_other):
        return self.value == coeff_other.value

    def __add__(self, coeff_other):
        # TODO: fix and return a new object
        self.value += coeff_other.value


class CoefficientManager:
    """
    Coefficient objcts manager

    Parameters
    ----------
        coefficient_config: dict
            dictionary with all the coefficients names and properties

    """

    def __init__(self, coefficient_config):
        self.elements = []
        for name, property_dict in coefficient_config.items():
            constrain = (
                property_dict["constrain"] if "constrain" in property_dict else None
            )
            self.elements.append(
                Coefficient(
                    name,
                    property_dict["min"],
                    property_dict["max"],
                    constrain=constrain,
                )
            )

    def __getattr__(self, attr):
        vals = []
        for obj in self.elements:
            vals.append(getattr(obj, attr))
        return vals

    def __getitem__(self, item):
        return self.elements[item]


# # TODO consider using a DataClass and always read coefficients properties
# # from this class and not from the config

# CoeffTuple = namedtuple("Coefficients", ("labels", "values", "bounds", "fixed"))


# def aggregate_coefficients(input_coefficients, loaded_datasets, input_bounds=None):
#     """
#     Aggregate all coefficient labels and construct an array of coefficient
#     values of suitable size. Returns a CoeffTuple of the labels, values,
#     and bounds

#     Parameters
#     ----------
#         config : dict
#             config dictionary
#         loaded_datasets : DataTuple
#             loaded datasets
#     Returns
#     -------
#         CT_CoeffTuple : CoeffTuple
#             CoeffTuple of the labels, values and bounds
#     """

#     # Give the initial point of the fit to be randomly spread around the bounds
#     # specified by --bounds option (if none given, bounds are taken from setup.py)
#     dataset_coefficients = []

#     # for set in loaded_datasets:
#     for key in loaded_datasets.CorrectionsKEYS:
#         dataset_coefficients.append(key)

#     # Keep ordering of coefficients the same so they match to the actual corrections
#     dataset_coefficients, idx = np.unique(
#         np.array(dataset_coefficients), return_index=True
#     )
#     dataset_coefficients = dataset_coefficients[np.argsort(idx)]

#     # All the coefficients are initialized to 0 by default
#     values = np.zeros(len(dataset_coefficients))
#     bounds = [(0.0, 0.0) for i in range(0, len(dataset_coefficients))]
#     fixed = [True for i in range(0, len(dataset_coefficients))]

#     # for k in config["coefficients"].keys():
#     for k in input_coefficients.keys():
#         if k not in dataset_coefficients:
#             raise ValueError(
#                 f"{k} is not part of fitted coefficients. Please comment it out in the setup file"
#             )

#         if input_bounds is None:
#             min_val = input_coefficients[k]["min"]
#             max_val = input_coefficients[k]["max"]

#         idx = np.where(dataset_coefficients == k)[0][0]
#         bounds[idx] = (min_val, max_val)
#         values[idx] = np.random.uniform(low=min_val, high=max_val)
#         fixed[idx] = input_coefficients[k]["fixed"]

#     return CoeffTuple(dataset_coefficients, values, bounds, fixed)
