import json
import numpy as np

from . import chi2 as chi2
from .loader import load_datasets 
from .loader import aggregate_coefficients



class OPTIMIZER:
    """
    Common interface for Chi2 profile, NS and McFiT

    Parameters
    ----------
        config : dict
            configuration dictionary
    """  
    def __init__(self, config):
        self.config = config
        self.loaded_datasets = load_datasets(self.config)
        self.coefficients = aggregate_coefficients(self.config, self.loaded_datasets)
    
        for k in self.config["coefficients"]:
            if k not in self.coefficients.labels:
                raise NotImplementedError(
                    f"{k} does not enter the theory. Comment it out in setup script and restart."
                )

        # Get indice locations for quadratic corrections
        if self.config["HOlambda"] == "HO":
            self.config["HOindex1"] = []
            self.config["HOindex2"] = []
            
            for coeff in self.loaded_datasets.HOcorrectionsKEYS:
                idx1 = np.where(self.coefficients.labels == coeff.split("*")[0])[0][0]
                idx2 = np.where(self.coefficients.labels == coeff.split("*")[1])[0][0]
                self.config["HOindex1"].append(idx1)
                self.config["HOindex2"].append(idx2)
            self.config["HOindex1"] = np.array(self.config["HOindex1"])
            self.config["HOindex2"] = np.array(self.config["HOindex2"])

        self.npts = None
        self.npts_V = None
        self.free_params = []
        self.free_param_labels = []


    def get_free_params(self):
        """Gets free parameters entering fit"""

        free_params = []
        free_param_labels = []
        for k in self.config["coefficients"].keys():

            idx = np.where(self.coefficients.labels == k)[0][0]
            if self.config["coefficients"][k]["fixed"] is False:
                free_params.append(self.coefficients.values[idx])
                free_param_labels.append(k)

        self.free_params = np.array(free_params)
        self.free_param_labels = np.array(free_param_labels)

    def set_constraints(self):
        """Sets parameter constraints"""
        for coeff_name, coeff in self.config["coefficients"].items():

            # free dof
            if coeff["fixed"] is False:
                continue

            idx = np.where(self.coefficients.labels == coeff_name)[0][0]
            # fixed to single value?
            if coeff["fixed"] is True:
                self.coefficients.values[idx] = coeff["value"]
                continue

            # # fixed to multiple values
            # rotation = np.array(coeff["value"])
            # new_post = []
            # for free_dof in coeff["fixed"]:
            #     idx_fixed = np.where(self.coefficients.labels == free_dof)[0][0]
            #     new_post.append(float(self.coefficients.values[idx_fixed]))

            # self.coefficients.values[idx] = rotation @ np.array(new_post)
    
    def propagate_params(self):
        """Propagates minimizer's updated parameters to the coefficient tuple"""

        for k in self.config["coefficients"]:
            idx = np.where(self.coefficients.labels == k)[0][0]
            if self.config["coefficients"][k]["fixed"] is False:
                idx2 = np.where(self.free_param_labels == k)[0][0]
                self.coefficients.values[idx] = self.free_params[idx2]
    
    def chi2_func(self):
        """
        Wrap the chi2 in a function for scipy optimiser. Pass noise and
        data info as args. Log the chi2 value and values of the coefficients.

        Returns
        -------
            current_chi2 : np.ndarray
                chi2 function
        """
        loaded_data = self.loaded_datasets
        coefficients = self.coefficients

        current_chi2, self.npts, self.npts_V = chi2.compute_total_chi2(
            self.config, loaded_data, coefficients.values, coefficients.labels
        )

        return current_chi2