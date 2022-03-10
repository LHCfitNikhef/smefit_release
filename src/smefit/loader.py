# -*- coding: utf-8 -*-

import pathlib
from collections import namedtuple

import numpy as np
import pandas as pd

from .covmat import build_large_covmat, construct_covmat


def check_file(path):
    """Check if path exists"""
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")


class Loader:
    """
    Class to check, load commondata and corresponding theory predictions

    Attributes
    ----------
        path: pathlib.path
            path to commondata folder, commondata excluded
        theory_path: pathlib.Path, optional
            path to theory folder, theory excluded.
            Default it assumes to be the same as path

    Parameters
    ----------
        setname: str
            dataset name to load
        operators_to_keep : list
            list of operators for which corrections are loaded
    """

    path = pathlib.Path()
    theory_path = pathlib.Path(path)
    _data_folder = pathlib.Path(path) / "commondata"
    _sys_folder = pathlib.Path(path) / "commondata/systypes"
    _theory_folder = pathlib.Path(theory_path) / "theory"

    def __init__(self, setname, operators_to_keep):
        self.setname = setname

        self.dataspec = {}
        (
            self.dataspec["central_values"],
            self.dataspec["covmat"],
        ) = self.load_experimental_data()
        (
            self.dataspec["SM_predictions"],
            self.dataspec["nho_corrections"],
            self.dataspec["ho_corrections"],
        ) = self.load_theory(operators_to_keep)

    def load_experimental_data(self):
        """
        Load experimental data with corresponding uncertainties

        Returns
        -------
            cental_values: numpy.ndarray
                experimental central values
            covmat : numpy.ndarray
                experimental covariance matrix
        """
        data_file = self._data_folder / f"DATA_{self.setname}.dat"
        sys_file = self._sys_folder / f"SYSTYPE_{self.setname}_DEFAULT.dat"

        check_file(data_file)
        check_file(sys_file)

        # load data from commondata file
        # TODO: better data format?
        # - DATA_* has many unused info
        num_data, num_sys = np.loadtxt(data_file, usecols=(1, 2), max_rows=1)
        central_values, stat_error = np.loadtxt(
            data_file, usecols=(5, 6), unpack=True, skiprows=1
        )

        # Load systematics from commondata file.
        # Read values of sys first
        sys_add = []
        sys_mult = []
        for i in range(0, num_sys):
            add, mult = np.loadtxt(
                data_file,
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
            sys_file,
            usecols=(1, 2),
            unpack=True,
            skip_header=1,
            dtype="str",
        )

        # Identify add and mult systematics
        # and replace the mult ones with corresponding value computed
        # from data central value. Required for implementation of t0 prescription

        indx_add = np.where(type_sys == "ADD")
        indx_mult = np.where(type_sys == "MULT")
        sys = np.zeros((num_sys, num_data))
        sys[indx_add[0]] = sys_add[indx_add[0]]
        sys[indx_mult[0]] = sys_mult[indx_mult[0]] * central_values * 1e-2

        # Build dataframe with shape (N_data * N_sys) and systematic name as the
        # column headers and construct covmat
        df = pd.DataFrame(data=sys, columns=name_sys)

        return central_values, construct_covmat(stat_error, df)

    def load_theory(self, operators_to_keep):
        """
        Load theory predictions

        Parameters
        ----------
            operators_to_keep: list
                list of operators to keep


        Returns
        -------
            sm: numpy.ndarray
                |SM| predictions
            nho_dict: dict
                dictionary with |NHO| corrections
            ho_dict: dict
                dictionary with |HO| corrections
        """
        theory_file = self._theory_folder / f"{self.setname}.txt"
        check_file(theory_file)

        # load theory predictions
        corrections_dict = {}
        with open(theory_file, encoding="utf-8") as f:
            for line in f:
                key, *val = line.split()
                corrections_dict[key] = np.array([float(v) for v in val])

        # Split the dictionary into SM, lambda^-2 and lambda^-4 terms
        # keep only needed corrections
        ho_dict = {}
        nho_dict = {}

        def is_to_keep(op1, op2=None):
            if op2 is None:
                return op1 in operators_to_keep
            return op1 in operators_to_keep and op2 in operators_to_keep

        for key, value in corrections_dict.items():
            if "*" in key:
                op1, op2 = key.split("*")
                if is_to_keep(op1, op2):
                    ho_dict[key] = value
            elif "^" in key:
                op = key[:-2]
                if is_to_keep(op):
                    new_key = f"{op}*{op}"
                    ho_dict[new_key] = value
            elif "SM" not in key:
                if is_to_keep(key):
                    nho_dict[key] = value

        # TODO: make sure we store theory and for EFT and SM in the same file,
        # for most of the old tables the things do not coincide
        return corrections_dict["SM"], nho_dict, ho_dict

    @property
    def n_data(self):
        """
        Number of data

        Returns:
        --------
            n_data: int
                number of experimental data
        """
        return self.dataspec["central_values"].size

    @property
    def central_values(self):
        """
        Central values

        Returns:
        --------
            central_values: numpy.ndarray
                experimental central values
        """
        return self.dataspec["central_values"]

    @property
    def covmat(self):
        """
        Covariance matrix

        Returns:
        --------
            covmat: numpy.ndarray
                experimental covariance matrix
        """
        return self.dataspec["covmat"]

    @property
    def sm_prediction(self):
        """
        |SM| prediction for the dataset

        Returns:
        --------
            SM_predictions : numpy.ndarray
                best |SM| prediction
        """
        return self.dataspec["SM_predictions"]

    @property
    def nho_corrections(self):
        """
        |NHO| corrections

        Returns:
        --------
            nho_corrections : dict
                dictionary with operator names and |NHO| correctsions
        """
        return self.dataspec["nho_corrections"]

    @property
    def ho_corrections(self):
        """
        |HO| corrections

        Returns:
        --------
            ho_corrections : dict
                dictionary with operator names and |HO| correctsions
        """
        return self.dataspec["ho_corrections"]


def split_corrections_dict(corrections_list, n_data_tot):
    """
    Construct a unique list of correction name,
    with corresponding values.

    Parameters
    ----------
        corrections_list : list(dict)
            list containing corrections per experiment
        n_data_tot : int
            total number of experimental datapoints

    Returns
    -------
        corr_keys : np.ndarray
            array containing correction names
        corr_values : np.ndarray
            matrix with correction values (n_data_tot,corr_keys.size)
    """

    corr_keys = np.unique(np.array([(*c,) for c in corrections_list]).flatten())
    corr_values = np.zeros((n_data_tot, corr_keys.size))

    cnt = 0
    # loop on experiments
    for correction_dict in corrections_list:
        for key, values in correction_dict.items():
            idx = np.where(corr_keys == key)[0][0]
            n_dat = values.size
            corr_values[cnt : cnt + n_dat, idx] = values

    return corr_keys, corr_values


# TODO: fix name convention, aggregate keys and vals?
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


def load_datasets(path, datasets, operators_to_keep):
    """
    Loads experimental data, theory and |SMEFT| corrections into a namedtuple

    Parameters
    ----------
        path : str
            root path
        datasets : list
            list of datasets to be loaded
        operators_to_keep: list
            list of operators for which corrections are loaded
    """

    exp_data = []
    sm_theory = []
    nho_corr_list = []
    ho_corr_list = []
    chi2_covmat = []
    n_data_exp = []
    exp_name = []

    Loader.path = path

    for sset in np.unique(datasets):

        dataset = Loader(sset, operators_to_keep)
        exp_name.append(sset)
        n_data_exp.append(dataset.n_data)

        exp_data.extend(dataset.central_values)
        sm_theory.extend(dataset.sm_prediction)

        nho_corr_list.append(dataset.nho_corrections)
        nho_corr_list.append(dataset.ho_corrections)
        chi2_covmat.append(dataset.covmat)

    exp_data = np.array(exp_data)
    n_data_tot = exp_data.size

    lin_corr_keys, lin_corr_values = split_corrections_dict(nho_corr_list, n_data_tot)
    quad_corr_keys, quad_corr_values = split_corrections_dict(ho_corr_list, n_data_tot)

    # Make one large datatuple containing all data, SM theory, corrections, etc.
    return DataTuple(
        exp_data,
        np.array(sm_theory),
        lin_corr_keys,
        lin_corr_values,
        quad_corr_keys,
        quad_corr_values,
        np.array(exp_name),
        np.array(n_data_exp),
        # Construct unique large cov matrix dropping correlations between different datasets
        build_large_covmat(chi2_covmat, n_data_tot, n_data_exp),
    )
