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
        return self.op_name == coeff_other.op_name

    def __lt__(self, coeff_other):
        return self.op_name < coeff_other.op_name

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
        # make sure elements are sorted by names
        self.elements = np.unique(self.elements)

    def __getattr__(self, attr):
        vals = []
        for obj in self.elements:
            vals.append(getattr(obj, attr))
        return np.array(vals)

    def __getitem__(self, item):
        return self.elements[item]
