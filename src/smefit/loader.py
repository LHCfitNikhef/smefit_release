# -*- coding: utf-8 -*-

import json
import pathlib
from collections import namedtuple

import numpy as np
import pandas as pd

from .basis_rotation import rotate_to_fit_basis
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
        commondata_path: pathlib.path
            path to commondata folder, commondata excluded
        theory_path: pathlib.Path, optional
            path to theory folder, theory excluded.
            Default it assumes to be the same as commondata_path

    Parameters
    ----------
        setname: str
            dataset name to load
        operators_to_keep : list
            list of operators for which corrections are loaded
        order: "LO", "NLO"
            EFT perturbative order
        use_quad : bool
            if True loads also |HO| corrections
        use_theory_covmat: bool
            if True add the theory covariance matrix to the experimental one
        rot_to_fit_basis: dict, None
            matrix rotation to fit basis or None
    """

    commondata_path = pathlib.Path()
    theory_path = pathlib.Path(commondata_path)

    def __init__(
        self,
        setname,
        operators_to_keep,
        order,
        use_quad,
        use_theory_covmat,
        rot_to_fit_basis,
    ):

        self._data_folder = self.commondata_path
        self._sys_folder = self.commondata_path / "systypes"
        self._theory_folder = self.theory_path

        self.setname = setname

        self.dataspec = {}
        (
            self.dataspec["central_values"],
            self.dataspec["covmat"],
        ) = self.load_experimental_data()
        (
            self.dataspec["SM_predictions"],
            self.dataspec["theory_covmat"],
            self.dataspec["lin_corrections"],
            self.dataspec["quad_corrections"],
        ) = self.load_theory(
            operators_to_keep, order, use_quad, use_theory_covmat, rot_to_fit_basis
        )

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
        num_data, num_sys = np.loadtxt(data_file, usecols=(1, 2), max_rows=1, dtype=int)
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
        df = pd.DataFrame(data=sys.T, columns=name_sys)
        return central_values, construct_covmat(stat_error, df)

    def load_theory(
        self,
        operators_to_keep,
        order,
        use_quad,
        use_theory_covmat,
        rotation_matrix=None,
    ):
        """
        Load theory predictions

        Parameters
        ----------
            operators_to_keep: list
                list of operators to keep
            order: "LO", "NLO"
                EFT perturbative order
            use_quad: bool
                if True returns also |HO| corrections
            use_theory_covmat: bool
                if True add the theory covariance matrix to the experimental one
            rotation_matrix: numpy.ndarray
                rotation matrix from tables basis to fitting basis

        Returns
        -------
            sm: numpy.ndarray
                |SM| predictions
            lin_dict: dict
                dictionary with |NHO| corrections
            quad_dict: dict
                dictionary with |HO| corrections, empty if not use_quad
        """
        theory_file = self._theory_folder / f"{self.setname}.json"
        check_file(theory_file)
        # load theory predictions
        with open(theory_file, encoding="utf-8") as f:
            raw_th_data = json.load(f)

        quad_dict = {}
        lin_dict = {}

        # split corrections into a linear and quadratic dict
        for key, value in raw_th_data[order].items():

            # quadratic terms
            if "*" in key and use_quad:
                quad_dict[key] = np.array(value)
            # linear terms
            elif "SM" not in key:
                lin_dict[key] = np.array(value)

        # rotate corrections to fitting basis
        if rotation_matrix is not None:
            lin_dict, quad_dict = rotate_to_fit_basis(
                lin_dict, quad_dict, rotation_matrix
            )

        # TODO: Move inside the rotation? can save some loops in case of rotation, but less clear
        # select corrections to keep
        def is_to_keep(op1, op2=None):
            if op2 is None:
                return op1 in operators_to_keep
            return op1 in operators_to_keep and op2 in operators_to_keep

        lin_dict_to_keep = {k: val for k, val in lin_dict.items() if is_to_keep(k)}
        quad_dict_to_keep = {
            k: val
            for k, val in quad_dict.items()
            if is_to_keep(k.split("*")[0], k.split("*")[1])
        }

        best_sm = np.array(raw_th_data["best_sm"])
        th_cov = np.zeros((best_sm.size, best_sm.size))
        if use_theory_covmat:
            th_cov = raw_th_data["theory_cov"]

        return raw_th_data["best_sm"], th_cov, lin_dict_to_keep, quad_dict_to_keep

    @property
    def n_data(self):
        """
        Number of data

        Returns
        -------
            n_data: int
                number of experimental data
        """
        return self.dataspec["central_values"].size

    @property
    def central_values(self):
        """
        Central values

        Returns
        -------
            central_values: numpy.ndarray
                experimental central values
        """
        return self.dataspec["central_values"]

    @property
    def covmat(self):
        """
        Covariance matrix

        Returns
        -------
            covmat: numpy.ndarray
                experimental covariance matrix
        """
        return self.dataspec["covmat"] + self.dataspec["theory_covmat"]

    @property
    def sm_prediction(self):
        """
        |SM| prediction for the dataset

        Returns
        -------
            SM_predictions : numpy.ndarray
                best |SM| prediction
        """
        return self.dataspec["SM_predictions"]

    @property
    def lin_corrections(self):
        """
        |NHO| corrections

        Returns
        -------
            lin_corrections : dict
                dictionary with operator names and |NHO| correctsions
        """
        return self.dataspec["lin_corrections"]

    @property
    def quad_corrections(self):
        """
        |HO| corrections

        Returns
        -------
            quad_corrections : dict
                dictionary with operator names and |HO| correctsions
        """
        return self.dataspec["quad_corrections"]


def construct_corrections_matrix(corrections_list, n_data_tot):
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
        sorted_keys : np.ndarray
            unique list of opearators for which at least one correction is present
        corr_values : np.ndarray
            matrix with correction values (n_data_tot, sorted_keys.size)
    """

    sorted_keys = np.unique(np.array([(*c,) for c in corrections_list]).flatten())
    corr_values = np.zeros((n_data_tot, sorted_keys.size))

    cnt = 0
    # loop on experiments
    for correction_dict in corrections_list:
        # loop on corrections
        for key, values in correction_dict.items():
            idx = np.where(sorted_keys == key)[0][0]
            n_dat = values.size
            corr_values[cnt : cnt + n_dat, idx] = values

    return sorted_keys, corr_values


DataTuple = namedtuple(
    "DataTuple",
    (
        "Commondata",
        "SMTheory",
        "OperatorsNames",
        "LinearCorrections",
        "QuadraticCorrections",
        "ExpNames",
        "NdataExp",
        "InvCovMat",
    ),
)


def load_datasets(
    commondata_path,
    datasets,
    operators_to_keep,
    order,
    use_quad,
    use_thory_covmat,
    theory_path=None,
    rot_to_fit_basis=None,
):
    """
    Loads experimental data, theory and |SMEFT| corrections into a namedtuple

    Parameters
    ----------
        commondata_path : str, pathlib.Path
            path to commondata folder, commondata excluded
        datasets : list
            list of datasets to be loaded
        operators_to_keep: list
            list of operators for which corrections are loaded
        order: "LO", "NLO"
            EFT perturbative order
        use_quad: bool
            if True loads also |HO| corrections
        use_theory_covmat: bool
            if True add the theory covariance matrix to the experimental one
        theory_path : str, pathlib.Path, optional
            path to theory folder, theory excluded.
            Default it assumes to be the same as commondata_path
        rot_to_fit_basis: dict, optional
            matrix rotation to fit basis or None
    """

    exp_data = []
    sm_theory = []
    lin_corr_list = []
    quad_corr_list = []
    chi2_covmat = []
    n_data_exp = []
    exp_name = []

    Loader.commondata_path = pathlib.Path(commondata_path)
    if theory_path is not None:
        Loader.theory_path = pathlib.Path(theory_path)
    else:
        Loader.theory_path = pathlib.Path(commondata_path)

    for sset in np.unique(datasets):

        dataset = Loader(
            sset, operators_to_keep, order, use_quad, use_thory_covmat, rot_to_fit_basis
        )
        exp_name.append(sset)
        n_data_exp.append(dataset.n_data)

        exp_data.extend(dataset.central_values)
        sm_theory.extend(dataset.sm_prediction)

        lin_corr_list.append(dataset.lin_corrections)
        quad_corr_list.append(dataset.quad_corrections)
        chi2_covmat.append(dataset.covmat)

    exp_data = np.array(exp_data)
    n_data_tot = exp_data.size

    operators_names, lin_corr_values = construct_corrections_matrix(
        lin_corr_list, n_data_tot
    )
    _, quad_corr_values = construct_corrections_matrix(quad_corr_list, n_data_tot)

    # Construct unique large cov matrix dropping correlations between different datasets
    covmat = (build_large_covmat(chi2_covmat, n_data_tot, n_data_exp),)

    # Make one large datatuple containing all data, SM theory, corrections, etc.
    return DataTuple(
        exp_data,
        np.array(sm_theory),
        operators_names,
        lin_corr_values,
        quad_corr_values,
        np.array(exp_name),
        np.array(n_data_exp),
        np.linalg.inv(covmat),
    )
