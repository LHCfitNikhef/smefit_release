# -*- coding: utf-8 -*-
"""Test utils module"""
import numpy as np

from smefit import coefficients

coeff_dict = {
    "op_a": {
        "min": -1,
        "max": 1,
    },
    "op_b": {
        "min": -2,
        "max": 1,
    },
}


class TestCoefficient:
    def test_init(self):
        name = "op_a"
        minimum = -1
        maximum = 1
        c_test = coefficients.Coefficient(name, minimum, maximum)
        assert c_test.op_name == name
        assert c_test.min == minimum
        assert c_test.max == maximum
        assert c_test.value <= maximum and c_test.value >= minimum

    def test_add(self):
        pass

    def test_eq(self):
        pass

    def test_repr(self):
        pass


class TestCoefficientManager:
    def test_init(self):
        c_list = coefficients.CoefficientManager(coeff_dict)

        # np.testing.assert_allclose(c_list.elements,["op_a", "op_b"])
        np.testing.assert_allclose(c_list.min, [-1, -2])
