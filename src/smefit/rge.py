import wilson
from smefit.wcxf import wcxf_translate, inverse_wcxf_translate
import smefit.log as log
from smefit.loader import Loader

import warnings
import pandas as pd
import numpy as np
import jax.numpy as jnp
from numpy import ComplexWarning
import pathlib

# Suppress the ComplexWarning
warnings.filterwarnings("ignore", category=ComplexWarning)

_logger = log.logging.getLogger(__name__)


class RGE:
    def __init__(self, wc_names, init_scale, accuracy="integrate"):
        # order the Wilson coefficients alphabetically
        self.wc_names = sorted(wc_names)
        self.init_scale = init_scale
        self.accuracy = accuracy

        _logger.info(
            f"Initializing RGE runner with initial scale {init_scale} GeV and accuracy {accuracy}."
        )

    def RGEmatrix_dict(self, scale):
        # compute the RGE matrix at the scale `scale`
        rge_matrix_dict = {}
        for wc_name, wc_vals in self.RGEbasis.items():
            _logger.info(f"Computing RGE for {wc_name} at {scale} GeV.")
            wc_init = wilson.Wilson(
                wc_vals, scale=self.init_scale, eft="SMEFT", basis="Warsaw"
            )
            wc_init.set_option("smeft_accuracy", self.accuracy)

            wc_final = wc_init.match_run(scale=scale, eft="SMEFT", basis="Warsaw")

            # Remove small values
            wc_final_vals = {
                key: value for key, value in wc_final.dict.items() if abs(value) > 1e-10
            }

            # check that imaginary values are small
            if any(abs(val.imag) > 1e-10 for val in wc_final_vals.values()):
                raise ValueError(
                    f"Imaginary values in Wilson coefficient for operator {wc_name}."
                )

            rge_matrix_dict[wc_name] = self.map_to_smefit(wc_final_vals)

        return rge_matrix_dict

    def RGEmatrix(self, scale):
        # compute the RGE matrix dict at the scale `scale`
        rge_matrix_dict = self.RGEmatrix_dict(scale)

        # create the RGE matrix as pandas dataframe
        rge_matrix = pd.DataFrame(
            columns=self.wc_names, index=self.all_ops, dtype=float
        )

        for wc_name, wc_dict in rge_matrix_dict.items():
            for op in self.all_ops:
                rge_matrix.loc[op, wc_name] = wc_dict.get(op, 0.0)
        # if there are rows with all zeros, remove them
        rge_matrix = rge_matrix.loc[(rge_matrix != 0).any(axis=1)]

        return rge_matrix

    @property
    def RGEbasis(self):
        # computes the translation from the smefit basis to the Warsaw basis
        # as expected by the Wilson package
        wc_basis = {}
        for wc_name in self.wc_names:
            try:
                wcxf_dict = wcxf_translate[wc_name]
            except KeyError:
                _logger.warning(
                    f"Wilson coefficient {wc_name} not present in the WCxf translation dictionary."
                )
                _logger.warning(
                    "Assuming it is a UV coupling and associating it to the null vector."
                )
                wc_basis[wc_name] = {}
                continue

            wc_warsaw_name = wcxf_dict["wc"]
            if "value" not in wcxf_dict:
                wc_warsaw_value = [1] * len(wcxf_dict["wc"])
            else:
                wc_warsaw_value = wcxf_dict["value"]

            # 1e-6 is because the Warsaw basis is in GeV^-2
            wc_value = {
                wc: val * 1e-6 for wc, val in zip(wc_warsaw_name, wc_warsaw_value)
            }
            wc_basis[wc_name] = wc_value

        return wc_basis

    def map_to_smefit(self, wc_final_vals):
        # TODO: missing a check that flavour structure is the one expected
        wc_dict = {}
        for wc_basis, wc_inv_dict in inverse_wcxf_translate.items():
            wc_warsaw_name = wc_inv_dict["wc"]
            if "coeff" not in wc_inv_dict:
                wc_warsaw_coeff = [1] * len(wc_warsaw_name)
            else:
                wc_warsaw_coeff = wc_inv_dict["coeff"]

            value = 0.0
            for wc, coeff in zip(wc_warsaw_name, wc_warsaw_coeff):
                if wc in wc_final_vals:
                    # 1e6 is to transform from GeV^-2 to TeV^2
                    value += 1e6 * wc_final_vals[wc].real * coeff
            wc_dict[wc_basis] = value
        return wc_dict

    @property
    def all_ops(self):
        return sorted(wcxf_translate.keys())

    def RGEevolve(self, wcs, scale):
        wc_wilson = {}
        for op, values in self.RGEbasis.items():
            for key in values:
                if key not in wc_wilson:
                    wc_wilson[key] = values[key] * wcs[op]
                else:
                    wc_wilson[key] += values[key] * wcs[op]

        wc_init = wilson.Wilson(
            wc_wilson, scale=self.init_scale, eft="SMEFT", basis="Warsaw"
        )
        wc_init.set_option("smeft_accuracy", self.accuracy)
        wc_final = wc_init.match_run(scale=scale, eft="SMEFT", basis="Warsaw").dict

        # remove small values
        wc_final = {key: value for key, value in wc_final.items() if abs(value) > 1e-10}

        return self.map_to_smefit(wc_final)


def load_scales(datasets, theory_path, default_scale=1e3):
    scales = []
    for dataset in np.unique(datasets):
        Loader.theory_path = pathlib.Path(theory_path)
        # dummy call just to get the scales
        _, _, _, _, dataset_scales = Loader.load_theory(
            dataset,
            operators_to_keep={},
            order="LO",
            use_quad=False,
            use_theory_covmat=False,
            use_multiplicative_prescription=False,
        )
        # check that dataset_scales is not a list filled with None
        # otherwise, assume the initial scale
        if not all([scale is None for scale in dataset_scales]):
            scales.extend(dataset_scales)
        else:
            scales.extend([default_scale] * len(dataset_scales))

        _logger.info(f"Loaded scales for dataset {dataset}: {dataset_scales}")

    return scales


def load_rge_matrix(rge_dict, operators_to_keep, datasets=None, theory_path=None):
    init_scale = rge_dict.get("init_scale", 1e3)
    obs_scale = rge_dict.get("obs_scale", 91.1876)
    smeft_accuracy = rge_dict.get("smeft_accuracy", "integrate")
    coeff_list = list(operators_to_keep.keys())
    rge_runner = RGE(coeff_list, init_scale, smeft_accuracy)
    # if it is a float, it is a static scale
    if type(obs_scale) is float or type(obs_scale) is int:
        rgemat = rge_runner.RGEmatrix(obs_scale)
        gen_operators = list(rgemat.index)
        operators_to_keep = {k: {} for k in gen_operators}
        return rgemat.values, operators_to_keep

    elif obs_scale == "dynamic":
        scales = load_scales(datasets, theory_path, default_scale=init_scale)

        operators_to_keep = {}
        rgemat = []
        rge_cache = {}
        for scale in scales:
            if scale == init_scale:
                # produce an identity matrix with row and columns coeff_list
                rgemat_scale = pd.DataFrame(
                    np.eye(len(coeff_list), len(coeff_list)),
                    columns=sorted(coeff_list),
                    index=sorted(coeff_list),
                )

            # Check if the RGE matrix has already been computed
            elif scale in rge_cache:
                rgemat_scale = rge_cache[scale]
            else:
                rgemat_scale = rge_runner.RGEmatrix(scale)
                gen_operators = list(rgemat_scale.index)
                # Fill with the operators if not already present in the dictionary
                for op in gen_operators:
                    if op not in operators_to_keep:
                        operators_to_keep[op] = {}
                rge_cache[scale] = rgemat_scale

            rgemat.append(rgemat_scale)

        # now loop through the rgemat and if there are operators
        # that are not present in the matrix, fill them with zeros
        for mat in rgemat:
            for op in operators_to_keep:
                if op not in mat.index:
                    mat.loc[op] = np.zeros(len(mat.columns))
            # order the rows alphabetically in the index
            mat.sort_index(inplace=True)

        # now stack the matrices in a 3D array
        stacked_mats = jnp.stack([mat.values for mat in rgemat])
        return stacked_mats, operators_to_keep

    else:
        raise ValueError(
            f"obs_scale must be either a float/int or 'dynamic'. Passed: {obs_scale}"
        )
