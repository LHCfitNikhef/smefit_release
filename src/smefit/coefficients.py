# -*- coding: utf-8 -*-

from tkinter.messagebox import NO

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
        constrain : dict, bool
            - if False, the parameter is free, default option
            - if True, the parameeter is fixed to the given value
            - if dict the parameter is fixed to a function of other coefficients

    """

    def __init__(self, name, minimum, maximum, value=None, constrain=False):
        self.op_name = name
        self.min = minimum
        self.max = maximum

        # determine if the parameter is free
        self.is_free = False
        self.constrain = None
        if constrain is False:
            if value is not None:
                raise ValueError(
                    f"Wilson Coefficient {self.op_name} is free, but a value is specified"
                )
            self.is_free = True
        elif constrain is True:
            if value is None:
                raise ValueError(
                    f"Wilson Coefficient {self.op_name} is fixed, but no value is specified"
                )
        elif isinstance(constrain, dict):

            self.constrain = constrain.copy()
            # update syntax for linear values
            for key, factor in constrain.items():
                if isinstance(factor, (int, float)):
                    self.constrain[key] = (factor, 1)

            raise ValueError(f"Unknown specified constrain {constrain}")

        # if no value is already there, the parameter is free
        self.value = value
        if value is None:
            self.value = np.random.uniform(low=minimum, high=maximum)

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
                property_dict["constrain"] if "constrain" in property_dict else False
            )
            self.elements.append(
                Coefficient(
                    name,
                    property_dict["min"],
                    property_dict["max"],
                    constrain=constrain,
                    value=property_dict["value"] if "value" in property_dict else None,
                )
            )
        # make sure elements are sorted by names
        self.elements = np.unique(self.elements)

    def __getattr__(self, attr):
        vals = []
        for obj in self.elements:
            vals.append(getattr(obj, attr))
        return np.array(vals)

    def get_from_name(self, item):
        """Return the list sliced by names"""
        return self.elements[self.op_name == item]

    def __getitem__(self, item):
        return self.elements[item]

    def free_parameters(self):
        """Returns the list containing only free parameters"""
        return self.elements[self.is_free]

    def set_constraints(self):
        r"""
        Sets constraints between coefficients:

        .. :math:
            c_{m} = \sum_{i=1} a_{i} c_{i}^{n_{i}}
        """

        # loop pn fixed coefficients
        for coefficient_fixed in self.elements[not self.is_free]:

            # skip coefficient fixed to a single value
            if coefficient_fixed.constrain is None:
                continue

            # fixed to multiple values
            constain_dict = coefficient_fixed.constrain
            free_dofs = self.get_from_name((*constain_dict,)).value

            # matrix with multiplicative factors and exponenets
            fact_exp = np.array((*constain_dict.values(),))

            self.get_from_name(coefficient_fixed).value = fact_exp[:, 0] @ np.power(
                free_dofs, fact_exp[:, 1]
            )
