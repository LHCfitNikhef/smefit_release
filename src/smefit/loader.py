# -*- coding: utf-8 -*-

import json
import pathlib
from collections import namedtuple

import numpy as np
import pandas as pd
import yaml

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
        data_file = self._data_folder / f"{self.setname}.yaml"
        check_file(data_file)

        print(f"Loaging datset : {self.setname}")
        with open(data_file, encoding="utf-8") as f:
            data_dict = yaml.safe_load(f)

        central_values = np.array(data_dict["data_central"])
        stat_error = np.array(data_dict["statistical_error"])

        num_sys = data_dict["num_sys"]
        num_data = data_dict["num_data"]

        # Load systematics from commondata file.
        # Read values of sys first

        sys_add = np.array(data_dict["systematics"])

        # Read systype file
        if num_sys != 0:
            type_sys = np.array(data_dict["sys_type"])
            name_sys = data_dict["sys_names"]

            # express systematics as percentage values of the central values
            sys_mult = sys_add / central_values * 1e2

            # Identify add and mult systematics
            # and replace the mult ones with corresponding value computed
            # from data central value. Required for implementation of t0 prescription

            indx_add = np.where(type_sys == "ADD")[0]
            indx_mult = np.where(type_sys == "MULT")[0]
            sys = np.zeros((num_sys, num_data))
            sys[indx_add] = sys_add[indx_add].reshape(sys[indx_add].shape)
            sys[indx_mult] = (
                sys_mult[indx_mult].reshape(sys[indx_mult].shape)
                * central_values
                * 1e-2
            )
            # Build dataframe with shape (N_data * N_sys) and systematic name as the
            # column headers and construct covmat

            # limit case with 1 sys
            if num_sys == 1:
                name_sys = [name_sys]
        # limit case no sys
        else:
            name_sys = ["UNCORR"]
            sys = np.zeros((num_sys + 1, num_data))

        df = pd.DataFrame(data=sys.T, columns=name_sys)

        # limit case 1 data
        if num_data == 1:
            central_values = np.asarray([central_values])

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
            elif "SM" not in key and "*" not in key:
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


def construct_corrections_matrix(corrections_list, n_data_tot, sorted_keys=None):
    """
    Construct a unique list of correction name,
    with corresponding values.

    Parameters
    ----------
        corrections_list : list(dict)
            list containing corrections per experiment
        n_data_tot : int
            total number of experimental data points
        sorted_keys: numpy.ndarray
            list of sorted operator corrections

    Returns
    -------
        sorted_keys : np.ndarray
            unique list of operators for which at least one correction is present
        corr_values : np.ndarray
            matrix with correction values (n_data_tot, sorted_keys.size)
    """

    if sorted_keys is None:
        tmp = [
            [
                *c,
            ]
            for _, c in corrections_list
        ]
        sorted_keys = np.unique([item for sublist in tmp for item in sublist])
    corr_values = np.zeros((n_data_tot, sorted_keys.size))
    cnt = 0
    # loop on experiments
    for n_dat, correction_dict in corrections_list:
        # loop on corrections
        for key, values in correction_dict.items():

            if "*" in key:
                op1, op2 = key.split("*")
                if op2 < op1:
                    key = f"{op2}*{op1}"

            idx = np.where(sorted_keys == key)[0][0]
            corr_values[cnt : cnt + n_dat, idx] = values
        cnt += n_dat

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
        "Replica",
    ),
)


def load_datasets(
    commondata_path,
    datasets,
    operators_to_keep,
    order,
    use_quad,
    use_theory_covmat,
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
            sset,
            operators_to_keep,
            order,
            use_quad,
            use_theory_covmat,
            rot_to_fit_basis,
        )
        exp_name.append(sset)
        n_data_exp.append(dataset.n_data)

        exp_data.extend(dataset.central_values)
        sm_theory.extend(dataset.sm_prediction)
        lin_corr_list.append([dataset.n_data, dataset.lin_corrections])
        quad_corr_list.append([dataset.n_data, dataset.quad_corrections])
        chi2_covmat.append(dataset.covmat)

    exp_data = np.array(exp_data)
    n_data_tot = exp_data.size

    operators_names, lin_corr_values = construct_corrections_matrix(
        lin_corr_list, n_data_tot
    )

    if use_quad:
        quad_corrections_names = []
        for op1 in operators_names:
            for op2 in operators_names:
                if (
                    f"{op1}*{op2}" not in quad_corrections_names
                    and f"{op2}*{op1}" not in quad_corrections_names
                ):
                    quad_corrections_names.append(f"{op1}*{op2}")
        _, quad_corr_values = construct_corrections_matrix(
            quad_corr_list, n_data_tot, np.array(quad_corrections_names)
        )
    else:
        quad_corr_values = None

    # Construct unique large cov matrix dropping correlations between different datasets
    covmat = build_large_covmat(chi2_covmat, n_data_tot, n_data_exp)
    replica = np.random.multivariate_normal(exp_data, covmat)
    # import pdb; pdb.set_trace()
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
        replica,
    )


def get_dataset(datasets, data_name):

    idx = np.where(datasets.ExpNames == data_name)[0][0]
    ndata = datasets.NdataExp[idx]
    posix_in = datasets.NdataExp[:idx].sum()
    posix_out = posix_in + ndata

    return DataTuple(
        datasets.Commondata[posix_in:posix_out],
        datasets.SMTheory[posix_in:posix_out],
        datasets.OperatorsNames,
        datasets.LinearCorrections[posix_in:posix_out],
        datasets.QuadraticCorrections[posix_in:posix_out]
        if datasets.QuadraticCorrections is not None
        else None,
        data_name,
        ndata,
        datasets.InvCovMat[posix_in:posix_out].T[posix_in:posix_out],
        datasets.Replica[posix_in:posix_out],
    )
