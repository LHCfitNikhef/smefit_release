# -*- coding: utf-8 -*-

import pathlib
import yaml
import numpy as np
import pandas as pd

from collections import namedtuple
from .covmat import construct_covmat
from .covmat import build_large_covmat


def check_file(path):
    """Check if path exists"""
    if not path.exists():
        raise FileNotFoundError(("File '%s' does not exist.") % (path))


class Loader:
    """
    Class to check and load commondata and corresponding theory predictions
    """

    def __init__(self, path, setname):
        self.setname = setname
        self.meta_folder = pathlib.Path(path) / f"commondata/meta/{self.setname}.yaml"
        self.data_folder = pathlib.Path(path) / f"commondata/DATA_{self.setname}.dat"
        self.sys_folder = (
            pathlib.Path(path)
            / f"commondata/systypes/SYSTYPE_{self.setname}_DEFAULT.dat"
        )
        self.theory_folder = pathlib.Path(path) / f"theory/{self.setname}.txt"

        self.dataspec = {}
        self.load_dataset()

    def load_dataset(self):
        """Load data, systematics and corresponding theory predictions"""

        # check that all relevant files exist
        check_file(self.meta_folder)
        check_file(self.data_folder)
        check_file(self.sys_folder)
        check_file(self.theory_folder)

        # load information from the meta file
        with open(f"{self.meta_folder}") as f:
            meta = yaml.safe_load(f)

        self.num_data = meta["npoints"]
        self.num_sys = meta["nsys"]

        # load data from commondata file
        central_values, stat_error = np.loadtxt(
            f"{self.data_folder}", usecols=(5, 6), unpack=True, skiprows=1
        )
        self.dataspec["central_values"] = central_values

        # Load systematics from commondata file.
        # Read values of sys first
        sys_add = list()
        sys_mult = list()
        for i in range(0, self.num_sys):
            add, mult = np.loadtxt(
                f"{self.data_folder}",
                usecols=(7 + 2 * i, 8 + 2 * i),
                unpack=True,
                skiprows=1,
            )
            sys_add.append(add)
            sys_mult.append(mult)

        sys_add = np.asarray(sys_add)
        sys_mult = np.asarray(sys_mult)

        # Read systype file
        type_sys, name_sys = np.genfromtxt(
            f"{self.sys_folder}",
            usecols=(1, 2),
            unpack=True,
            skip_header=1,
            dtype="str",
        )

        # Identify add and mult systematics and replace the mult ones with corresponding value computed
        # from data central value. Required for implementation of t0 prescription

        indx_add = np.where(type_sys == "ADD")
        indx_mult = np.where(type_sys == "MULT")
        sys = np.zeros((self.num_sys, self.num_data))
        sys[indx_add[0]] = sys_add[indx_add[0]]
        sys[indx_mult[0]] = sys_mult[indx_mult[0]] * central_values * 1e-2

        # Build dataframe with shape (N_data * N_sys) and systematic name as the
        # column headers and construct covmat
        df = pd.DataFrame(data=sys, columns=name_sys)
        self.dataspec["covmat"] = construct_covmat(stat_error, df)

        # load theory predictions
        corrections_dic = {}
        with open(self.theory_folder) as f:
            for line in f:
                key, *val = line.split()
                corrections_dic[key] = np.array(
                    [float(val[i]) for i in range(len(val))]
                )
                # save sm predictions in a vectore and remove from the dictionary
        self.dataspec["SM_predictions"] = corrections_dic["SM"]
        corrections_dic.pop("SM", None)

        # Split the dictionary into lambda^-2 and lambda^-4 terms
        higherorder = {}
        removeHO = list()

        for key, value in corrections_dic.items():
            if "*" in key:
                removeHO.append(key)
                higherorder[key] = value
            if "^" in key:
                new_key = "%s*%s" % (key[:-2], key[:-2])
                removeHO.append(key)
                higherorder[new_key] = value

        for key in removeHO:
            if key in corrections_dic:
                del corrections_dic[key]

        self.dataspec["linear_corrections"] = corrections_dic
        self.dataspec["quadratic_corrections"] = higherorder

    def get_setname(self):
        """Get name of the dataset"""
        return self.setname

    def get_Ndata(self):
        """Get number of data"""
        return self.num_data

    def get_central_values(self):
        """Get central values"""
        return self.dataspec["central_values"]

    def get_covmat(self):
        """Get central values"""
        return self.dataspec["covmat"]

    def get_sm_prediction(self):
        """Return SM prediction for the dataset"""
        return self.dataspec["SM_predictions"]

    def get_linear_corrections(self):
        """Return linear corrections, as a dictionary with name of correction and its value"""
        return self.dataspec["linear_corrections"]

    def get_quadratic_corrections(self):
        """Return quadratic corrections, as a dictionary with name of correction and its value"""
        return self.dataspec["quadratic_corrections"]


DataTuple = namedtuple(
    "DataTuple",
    (
        "Commondata",
        "SMTheory",
        "CorrectionsKEYS",
        "CorrectionsVAL",
        "HOcorrectionsKEYS",
        "HOcorrectionsVAL",
        "ExpNames",
        "NdataExp",
        "CovMat",
    ),
)

#TODO: fix names convention
def load_datasets(config):
    """
    Loads commondata, theory and SMEFT corrections into a namedtuple

    Parameters
    ----------
        config : dict
            configuration dictionary
    """

    EXP_DATA = []
    SM_THEORY = []
    LIN_DICT = []
    QUAD_DICT = []
    CHI2_COVMAT = []
    N_data_exp = []
    EXP_name = []

    datasets = config["datasets"]
    path = config["root_path"]

    for sset in datasets:

        dataset = Loader(path, sset)
        EXP_name.append(sset)
        N_data_exp.append(dataset.get_Ndata())
        EXP_DATA.append(dataset.get_central_values())
        SM_THEORY.append(dataset.get_sm_prediction())
        LIN_DICT.append(dataset.get_linear_corrections())
        QUAD_DICT.append(dataset.get_quadratic_corrections())
        CHI2_COVMAT.append(dataset.get_covmat())

    # Flatten
    EXP_names = np.array(EXP_name)  # array containing all the datasets names
    EXP_array = flatten(EXP_DATA)  # array containing all data
    SM_array = flatten(SM_THEORY)  # array containing all SM theory

    # Construct unique large cov matrix dropping correlations between different datasets
    ndata = len(EXP_array)
    N_data_exp = np.array(N_data_exp)
    COVMAT_array = build_large_covmat(ndata, CHI2_COVMAT, N_data_exp)

    lin_corr_keys, lin_corr_values = split_corrections_dict(LIN_DICT, ndata)
    quad_corr_keys, quad_corr_values = split_corrections_dict(QUAD_DICT, ndata)

    # Make one large datatuple containing all data, SM theory, corrections, etc.
    return DataTuple(
        EXP_array,
        SM_array,
        lin_corr_keys,
        lin_corr_values,
        quad_corr_keys,
        quad_corr_values,
        EXP_names,
        N_data_exp,
        COVMAT_array,
    )


def flatten(input_dict):
    """
    Flatten a dictionary per experiment into a single array

    Parameters
    ----------
        in_dict : dict
            array containing values per experiment

    Returns
    -------
        np.ndarray
            arrays containing values
    """
    return np.array([item for sublist in input_dict for item in sublist])

#TODO: split this function and simplify
def split_corrections_dict(corrections_dict, ndata):
    """
    Store keys for correction values and build matrix containing
    the corrections for each coefficient

    Parameters
    ----------
        corrections_dict : dict
            array containing corrections per experiment
        ndata : int
            number of experimental datapoints

    Returns
    -------
        keys, vals : np.ndarray, np.ndarray
            arrays containing keys (experiments) and corrections
    """

    array = [item for sublist in corrections_dict for item in sublist]
    corr_keys = np.array(sorted(set(array)))

    corr_values = np.zeros((ndata, len(corr_keys)))

    if corr_keys.size == 0:
        return corr_keys, corr_values
    else:
        cnt = 0
        for correction_dict in corrections_dict:
            dummy_key = list(correction_dict.keys())[0]
            for j in range(len(correction_dict[dummy_key])):
                for key in correction_dict:
                    idx = np.where(corr_keys == key)[0][0]
                    corr_values[cnt, idx] = float(correction_dict[key][j])
                cnt += 1
        return corr_keys, corr_values

#TODO: consider using a DataClass and always read coefficients properties
# from this class and not from the config

CoeffTuple = namedtuple("Coefficients", ("labels", "values", "bounds"))


def aggregate_coefficients(config, loaded_datasets):
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
    coeff_labels = []
    bounds = {}

    # for set in loaded_datasets:
    for key in loaded_datasets.CorrectionsKEYS:
        coeff_labels.append(key)

    # Keep ordering of coefficients the same so they match to the actual corrections
    coeff_labels, idx = np.unique(np.array(coeff_labels), return_index=True)
    coeff_labels = coeff_labels[np.argsort(idx)]

    # All the coefficients are initialized to 0 by default
    randarr = np.zeros(len(coeff_labels))

    for k in config["coefficients"].keys():
        if k not in coeff_labels:
            raise ValueError(
                f"{k} is not part of fitted coefficients. Please comment it out in the setup file"
            )
        if config["coefficients"][k]["fixed"]:
            continue

        if config["bounds"] is None:
            min_val = config["coefficients"][k]["min"]
            max_val = config["coefficients"][k]["max"]

        idx = np.where(coeff_labels == k)[0][0]
        randarr[idx] = np.random.uniform(low=min_val, high=max_val)
        bounds[k] = (min_val, max_val)

    return CoeffTuple(coeff_labels, randarr, bounds)
