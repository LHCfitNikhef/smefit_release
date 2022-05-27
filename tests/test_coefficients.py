# -*- coding: utf-8 -*-
"""Test utils module"""
import numpy as np

from smefit import coefficients

coeff_dict = {
    "op_b": {
        "min": -1,
        "max": 1,
    },
    "op_a": {
        "min": -2,
        "max": 1,
    },
    "op_c": {
        "constrain": True,
        "value": 1,
        "min": -2,
        "max": 1,
    },
}


class TestCoefficient:

    name = "op_a"
    minimum = -1
    maximum = 1
    c_test = coefficients.Coefficient(name, minimum, maximum)

    def test_init(self):

        assert self.c_test.op_name == self.name
        assert self.c_test.minimum == self.minimum
        assert self.c_test.maximum == self.maximum
        assert self.c_test.value <= self.maximum and self.c_test.value >= self.minimum

    def test_add(self):
        rand_val_2 = np.random.rand()
        rand_val_1 = self.c_test.value
        c_2 = coefficients.Coefficient(
            "op_a", np.random.rand(), np.random.rand(), value=rand_val_2, constrain=True
        )
        self.c_test += c_2
        assert self.c_test.value == rand_val_1 + rand_val_2

    def test_eq(self):
        c_same = coefficients.Coefficient("op_a", np.random.rand(), np.random.rand())
        assert self.c_test == c_same

    def test_lt(self):
        c_same = coefficients.Coefficient("op_a", np.random.rand(), np.random.rand())
        assert not self.c_test < c_same
        c_diff = coefficients.Coefficient("op_c", np.random.rand(), np.random.rand())
        assert self.c_test < c_diff

    def test_repr(self):
        assert repr(self.c_test) == self.name


class TestCoefficientManager:

    c_list = coefficients.CoefficientManager.from_dict(coeff_dict)

    def test_init(self):

        # np.testing.assert_allclose(self.c_list.elements,["op_a", "op_b"])
        assert self.c_list[0].op_name == "op_a"
        np.testing.assert_allclose(self.c_list.minimum, [-2, -1, -2])
