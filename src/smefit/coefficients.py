# -*- coding: utf-8 -*-

from typing import Iterable

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
        self.minimum = minimum
        self.maximum = maximum

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
                    self.constrain[key] = np.array([factor, 1])
                else:
                    factor = np.array(factor)
                    # if factor has not 2 elements or is not a number raise error
                    if factor.size > 2 or factor.dtype not in [int, float]:
                        raise ValueError(f"Unknown specified constrain {constrain}")
                    self.constrain[key] = factor

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
        self.value += coeff_other.value
        return self


class CoefficientManager(np.ndarray):
    """
    Coefficient objcts manager

    Parameters
    ----------
        input_array: np.ndarray or list
            list of `smefit.coefficients.Coefficient` instances

    """

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array, dtype=object).view(cls)
        # add the new attribute to the created instance
        # obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return

    @classmethod
    def from_dict(cls, coefficient_config):
        """
        Create a coefficientManager from a dictionary

        Parmeters
        ---------
            coefficient_config : dict
                coefficients configuration dictionary

        Returns
        -------
            coefficient_manager: `smefit.coefficients.CoefficientManager`
                instance of the class
        """
        elements = []
        for name, property_dict in coefficient_config.items():
            constrain = (
                property_dict["constrain"] if "constrain" in property_dict else False
            )
            elements.append(
                Coefficient(
                    name,
                    property_dict["min"],
                    property_dict["max"],
                    constrain=constrain,
                    value=property_dict["value"] if "value" in property_dict else None,
                )
            )
        # make sure elements are sorted by names
        return cls(np.unique(elements))

    def __getattr__(self, attr):
        vals = []
        for obj in self:
            vals.append(getattr(obj, attr))
        return np.array(vals)

    def __setattr__(self, attr, value):
        if not isinstance(value, Iterable):
            value = [value]
        for obj, val in zip(self, value):
            setattr(obj, attr, val)

    def get_from_name(self, item):
        """Return the class sliced by names"""
        return self[self.op_name == item]

    @property
    def free_parameters(self):
        """Returns the class containing only free parameters"""
        return self[self.is_free]

    def set_constraints(self):
        r"""
        Sets constraints between coefficients according to the
        coefficient.constrain information:

        .. :math:
            c_{m} = \sum_{i=1} a_{i} c_{i}^{n_{i}}
        """

        # loop pn fixed coefficients
        for coefficient_fixed in self[np.invert(self.is_free)]:

            # skip coefficient fixed to a single value
            if coefficient_fixed.constrain is None:
                continue

            # fixed to multiple values
            constrain_dict = coefficient_fixed.constrain
            free_dofs = self.get_from_name((*constrain_dict,)).value

            # matrix with multiplicative factors and exponents
            fact_exp = np.array((*constrain_dict.values(),))

            self.get_from_name(coefficient_fixed.op_name).value = fact_exp[
                :, 0
            ] @ np.power(free_dofs, fact_exp[:, 1])
