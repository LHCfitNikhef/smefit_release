# -*- coding: utf-8 -*-
import importlib
import json
import pathlib

import numpy as np
from rich.style import Style
from rich.table import Table

from smefit.rge import RGE

from .. import chi2, log
from ..coefficients import CoefficientManager
from ..loader import get_dataset

try:
    from mpi4py import MPI

    run_parallel = True
except ModuleNotFoundError:
    run_parallel = False

_logger = log.logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.float32, np.float64, np.int32, np.int64)):
            return o.item()  # Convert to native Python type
        return super().default(o)


class Optimizer:
    """
    Common interface for Chi2 profile, NS and MC and A optimizers.

    Parameters
    ----------
        results_path : pathlib.path
            path to result folder
        loaded_datasets : DataTuple,
            dataset tuple
        coefficients : `smefit.coefficients.CoefficientManager`
            instance of `CoefficientManager` with all the relevant coefficients to fit
        use_quad : bool
            if True includes also |HO| correction
        single_parameter_fits : bool
            True for single parameter fits
        use_multiplicative_prescription:
            if True uses the multiplicative prescription for the |EFT| corrections.
        external_chi2: dict
            dict of external chi2
        rgemat: numpy.ndarray
            solution matrix of the RGE
        rge_dict: dict
            dictionary with the RGE input parameter options
    """

    print_rate = 500

    def __init__(
        self,
        results_path,
        loaded_datasets,
        coefficients,
        use_quad,
        single_parameter_fits,
        use_multiplicative_prescription,
        external_chi2=None,
        rgemat=None,
        rge_dict=None,
    ):
        self.results_path = pathlib.Path(results_path)
        self.loaded_datasets = loaded_datasets
        self.coefficients = coefficients
        self.use_quad = use_quad
        self.npts = (
            self.loaded_datasets.Commondata.size
            if self.loaded_datasets is not None
            else 0
        )
        self.single_parameter_fits = single_parameter_fits
        self.use_multiplicative_prescription = use_multiplicative_prescription
        self.counter = 0

        # set RGE matrix
        self.rgemat = rgemat
        self.rge_dict = rge_dict

        # load external chi2 modules as amortized objects (fast to evaluate)
        self.chi2_ext = (
            self.load_external_chi2(external_chi2) if external_chi2 else None
        )

    def load_external_chi2(self, external_chi2):
        """
        Loads the external chi2 modules

        Parameters
        ----------
        external_chi2: dict
            dict of external chi2s, with the name of the function object as key and the path to the external script
            as value

        Returns
        -------
        ext_chi2_modules: list
             List of external chi2 objects that can be evaluated by passing a coefficients instance
        """
        # dynamical import
        ext_chi2_modules = []

        for ext_likelihood_dict in external_chi2:
            path = pathlib.Path(ext_likelihood_dict["path"])
            ext_likelihood_mod = importlib.import_module(path.stem)
            my_ext_likelihood = getattr(
                ext_likelihood_mod, ext_likelihood_dict["likelihood_type"]
            )
            likelihood_ext = my_ext_likelihood(self.coefficients, ext_likelihood_dict)
            ext_chi2_modules.append(likelihood_ext.compute_chi2)

        # for class_name, module in external_chi2.items():
        #     module_path = module["path"]
        #     path = pathlib.Path(module_path)
        #     base_path, stem = path.parent, path.stem
        #     chi2_module = importlib.import_module(stem)
        #
        #     my_chi2_class = getattr(chi2_module, class_name)
        #
        #     if self.rge_dict is not None:
        #         # Check if dynamic scale
        #         if self.rge_dict["obs_scale"] == "dynamic":
        #             _logger.info(
        #                 f"Computing RGE matrix for {class_name} "
        #                 f"with initial scale {self.rge_dict['init_scale']}."
        #             )
        #             # compute RGE matrix
        #             if "scale" not in module:
        #                 raise ValueError(
        #                     "Dynamic scale requested but no scale provided in the external chi2"
        #                 )
        #             scale = module["scale"]
        #             rge_runner = RGE(
        #                 self.coefficients.name,
        #                 self.rge_dict["init_scale"],
        #                 self.rge_dict.get("smeft_accuracy", "integrate"),
        #             )
        #             rge_df = rge_runner.RGEmatrix(scale)
        #             gen_operators = list(rge_df.index)
        #             _logger.info("The operators generated by the RGE are: ")
        #             _logger.info(gen_operators)
        #             operators_dict = {
        #                 k: {"max": 0.0, "min": 0.0} for k in gen_operators
        #             }
        #             new_coeffs = CoefficientManager.from_dict(operators_dict)
        #             chi2_ext = my_chi2_class(new_coeffs, rge_df.values)
        #         else:
        #             gen_operators = list(self.loaded_datasets.OperatorsNames)
        #             # Create dummy coefficients
        #             operators_dict = {
        #                 k: {"max": 0.0, "min": 0.0} for k in gen_operators
        #             }
        #             new_coeffs = CoefficientManager.from_dict(operators_dict)
        #             chi2_ext = my_chi2_class(new_coeffs, self.rgemat)
        #     else:
        #         chi2_ext = my_chi2_class(self.coefficients)
        #
        #     ext_chi2_modules.append(chi2_ext.compute_chi2)

        return ext_chi2_modules

    @property
    def free_parameters(self):
        """Returns the free parameters entering fit"""
        return self.coefficients.free_parameters

    def generate_chi2_table(self, chi2_dict, chi2_tot):
        r"""Generate log :math:`\chi^2` table"""
        table = Table(style=Style(color="white"), title_style="bold cyan", title=None)
        table.add_column("Dataset", style="bold green", no_wrap=True)

        table.add_column("Chi^2 /N_dat")
        for name, val in chi2_dict.items():
            table.add_row(str(name), f"{val:.5}")
        table.add_row("Total", f"{(chi2_tot/self.npts):.5}")

        return table

    def chi2_func(self, use_replica=False, print_log=True):
        r"""
        Wrap the math:`\chi^2` in a function for the optimizer. Pass noise and
        data info as args. Log the math:`\chi^2` value and values of the coefficients.

        Returns
        -------
            current_chi2 : np.ndarray
                computed :math:`\chi^2`
        """
        rank = 0
        if run_parallel:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

        if rank == 0:
            self.counter += 1
            if print_log:
                print_log = (self.counter % self.print_rate) == 0
        else:
            print_log = False

        # only compute the internal chi2 when datasets are loaded
        if self.loaded_datasets is not None:
            chi2_tot = chi2.compute_chi2(
                self.loaded_datasets,
                self.coefficients.value,
                self.use_quad,
                self.use_multiplicative_prescription,
                use_replica,
            )
        else:
            chi2_tot = 0

        if self.chi2_ext is not None:
            for chi2_ext in self.chi2_ext:
                chi2_ext_i = chi2_ext(self.coefficients.value)
                chi2_tot += chi2_ext_i

        if print_log:
            chi2_dict = {}
            for data_name in self.loaded_datasets.ExpNames:
                dataset = get_dataset(self.loaded_datasets, data_name)
                chi2_dict[data_name] = (
                    chi2.compute_chi2(
                        dataset,
                        self.coefficients.value,
                        self.use_quad,
                        self.use_multiplicative_prescription,
                        use_replica,
                    )
                    / dataset.NdataExp
                )
            log.console.print(self.generate_chi2_table(chi2_dict, chi2_tot))

        return chi2_tot

    def dump_fit_result(self, fit_result_file, values):
        """
        Dumps the fit results to a json file.

        dump_fit_result gets called repeatedly for single parameter fits, once for each parameter.
        `values` contains the samples of the current fit, while previous fit results get loaded
        into `tmp` and updated with the current samples. The updated values are then written back
        to the file.

        Parameters
        ----------
        fit_result_file: PosixPath
            path to the fit results file
        values: dict
            dictionary containing the current fit results
        """

        if self.single_parameter_fits:
            if fit_result_file.is_file():
                with open(fit_result_file, encoding="utf-8") as f:
                    tmp = json.load(f)
                    # Get the operator name
                    coeff = list(values["samples"].keys())[0]
                    # update the values
                    tmp["logz"][coeff] = values["logz"]
                    tmp["max_loglikelihood"][coeff] = values["max_loglikelihood"]
                    tmp["best_fit_point"][coeff] = values["best_fit_point"][coeff]
                    tmp["samples"][coeff] = values["samples"][coeff]
                    # update the file with the new values
                    with open(fit_result_file, "w", encoding="utf-8") as f:
                        json.dump(tmp, f, indent=4, cls=NumpyEncoder)

            else:
                values["single_parameter_fits"] = True
                with open(fit_result_file, "w", encoding="utf-8") as f:
                    # Get the operator name
                    coeff = list(values["best_fit_point"].keys())[0]
                    values["logz"] = {coeff: values["logz"]}
                    values["max_loglikelihood"] = {coeff: values["max_loglikelihood"]}
                    json.dump(values, f, indent=4, cls=NumpyEncoder)

        else:
            with open(fit_result_file, "w", encoding="utf-8") as f:
                json.dump(values, f, indent=4, cls=NumpyEncoder)
