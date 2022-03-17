# -*- coding: utf-8 -*-
import pathlib

from .. import chi2


class Optimizer:
    """
    Common interface for Chi2 profile, NS and McFiT

    Parameters
    ----------
        results_path : pathlib.path

        loaded_datasets : DataTuple,
            dataset tuple
        coefficients :

        use_quad :
            if True include also |HO| correction

    """

    # TODO: docstring

    def __init__(self, results_path, loaded_datasets, coefficients, use_quad):
        self.results_path = pathlib.Path(results_path)
        self.loaded_datasets = loaded_datasets
        self.coefficients = coefficients
        self.use_quad = use_quad
        self.npts = self.loaded_datasets.Commondata.size

    @property
    def free_parameters(self):
        """Returns the free parameters entering fit"""
        return self.coefficients.free_parameters

    def chi2_func(self):
        r"""
        Wrap the math:`\Chi^2` in a function for scipy optimizer. Pass noise and
        data info as args. Log the math:`\Chi^2` value and values of the coefficients.

        Returns
        -------
            current_chi2 : np.ndarray
                computed :math:`\Chi^2`
        """
        return chi2.compute_chi2(
            self.loaded_datasets,
            self.coefficients.value,
            self.use_quad,
        )
