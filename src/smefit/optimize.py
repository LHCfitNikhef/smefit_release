# -*- coding: utf-8 -*-

import numpy as np

from . import chi2


class Optimizer:
    """
    Common interface for Chi2 profile, NS and McFiT

    Parameters
    ----------

    """

    def __init__(self, loaded_datasets, coefficients, HOindices):

        self.loaded_datasets = loaded_datasets
        self.coefficients = coefficients
        self.HOindices = HOindices
        self.npts = self.loaded_datasets.Commondata.size
        self.free_params = {}

    def get_free_params(self):
        """Gets free parameters entering fit"""

        for index, k in enumerate(self.coefficients.fixed):
            if k is False:
                free_coefficient_label = self.coefficients.labels[index]
                free_coefficient_value = self.coefficients.values[index]
                self.free_params[free_coefficient_label] = free_coefficient_value

    # #TODO Is this function necessary now?
    # def set_constraints(self):
    #     """Sets parameter constraints"""
    #     for coeff_name, coeff in self.config["coefficients"].items():

    #         # free dof
    #         if coeff["fixed"] is False:
    #             continue

    #         idx = np.where(self.coefficients.labels == coeff_name)[0][0]
    #         # fixed to single value?
    #         if coeff["fixed"] is True:
    #             self.coefficients.values[idx] = coeff["value"]
    #             continue

    #         # # fixed to multiple values
    #         # rotation = np.array(coeff["value"])
    #         # new_post = []
    #         # for free_dof in coeff["fixed"]:
    #         #     idx_fixed = np.where(self.coefficients.labels == free_dof)[0][0]
    #         #     new_post.append(float(self.coefficients.values[idx_fixed]))

    #         # self.coefficients.values[idx] = rotation @ np.array(new_post)

    def propagate_params(self):
        """Propagates minimizer's updated parameters to the coefficient tuple"""

        for idx, coefficient in enumerate(self.coefficients.labels):
            if self.coefficients.fixed[idx] is False:
                self.coefficients.values[idx] = self.free_params[coefficient]

    def chi2_func(self):
        """
        Wrap the chi2 in a function for scipy optimiser. Pass noise and
        data info as args. Log the chi2 value and values of the coefficients.

        Returns
        -------
            current_chi2 : np.ndarray
                chi2 function
        """

        return chi2.compute_chi2(
            self.loaded_datasets,
            self.coefficients.values,
            self.coefficients.labels,
            self.HOindices,
        )
