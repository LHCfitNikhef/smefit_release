# -*- coding: utf-8 -*-
import pathlib
import subprocess
from shutil import copyfile

import yaml
from mpi4py import MPI

from .chi2 import Scanner
from .log import logging
from .optimize.mc import MCOptimizer
from .optimize.ns import NSOptimizer

_logger = logging.getLogger(__name__)


class Runner:
    """
    Container for all the possible |SMEFiT| run methods.

    Init the root path of the package where tables,
    results, plot config and reports are stored.

    Parameters
    ----------
        run_card : dict
            run card dictionary
        runcard_folder: pathlib.Path, None
            path to runcard folder if already present
    """

    def __init__(self, run_card, single_parameter_fits, runcard_folder=None):

        self.run_card = run_card
        self.runcard_folder = runcard_folder
        self.single_parameter_fits = single_parameter_fits
        self.setup_result_folder()

    def setup_result_folder(self):
        """
        Create result folder and copy the runcard there
        """
        # Construct results folder
        run_card_name = self.run_card["runcard_name"]
        run_card_id = self.run_card["result_ID"]
        result_folder = pathlib.Path(self.run_card["result_path"])
        if self.run_card["replica"] is not None:
            res_folder_fit = (
                result_folder / run_card_id / f"replica_{self.run_card['replica']}"
            )
        else:
            res_folder_fit = result_folder / run_card_id

        subprocess.call(f"mkdir -p {result_folder}", shell=True)
        if res_folder_fit.exists():
            _logger.warning(f"{res_folder_fit} already found, overwriting old results")
        subprocess.call(f"mkdir -p {res_folder_fit}", shell=True)

        # Copy yaml runcard to results folder or dump it
        # in case no given file is passed
        runcard_copy = result_folder / run_card_id / f"{run_card_id}.yaml"
        if self.runcard_folder is None:
            with open(runcard_copy, encoding="utf-8") as f:
                yaml.dump(self.run_card, f, default_flow_style=False)
        else:
            copyfile(
                self.runcard_folder / f"{run_card_name}.yaml",
                runcard_copy,
            )

    @classmethod
    def from_file(cls, runcard_folder, run_card_name, replica=None):
        """
        Create Runner from a runcard file

        Parameters
        ----------
        runcard_folder: pathlib.Path, str
            path to runcard folder if already present
        run_card_name : srt
            run card name
        replica: int
            replica number. Optional used only for MC

        Returns
        -------
            runner: `smefit.runner.Runner`
                instance of class Runner
        """
        config = {}
        # load file
        with open(runcard_folder / f"{run_card_name}.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        config["runcard_name"] = run_card_name
        config["replica"] = replica
        # set result ID to runcard name by default
        if "result_ID" not in config:
            config["result_ID"] = run_card_name

        single_parameter_fits = config.get("single_parameter_fits", False)

        return cls(config, single_parameter_fits, runcard_folder)

    def ns(self, config):
        """Run a fit with |NS|."""

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            opt = NSOptimizer.from_dict(config)
        else:
            opt = None

        # Run optimizer
        opt = comm.bcast(opt, root=0)
        opt.run_sampling()

    def mc(self, config):
        """Run a fit with |MC|."""
        config = self.run_card
        opt = MCOptimizer.from_dict(config)
        opt.run_sampling()
        opt.save()

    def global_analysis(self, optimizer):
        """Run a global fit using the selected optimizer
        Parameters
        ----------
        optimizer: string
            optimizer to be used (NS or MC)
        """

        config = self.run_card
        if optimizer == "NS":
            self.ns(config)
        elif optimizer == "MC":
            self.mc(config)

    def single_parameter_analysis(self, optimizer):
        """Run a seried of single parameter fits for all the operators specified in the runcard
        Parameters
        ----------
        optimizer: string
            optimizer to be used (NS or MC)
        """

        config = self.run_card
        for coeff in config["coefficients"].keys():
            single_coeff_config = dict(config)
            single_coeff_config["coefficients"] = {}
            single_coeff_config["coefficients"][coeff] = config["coefficients"][coeff]

            if optimizer == "NS":
                self.ns(single_coeff_config)
            elif optimizer == "MC":
                self.mc(single_coeff_config)

    def run_analysis(self, optimizer):
        """Run either the global analysis or a series of single parameter fits
        using the selected optimizer.
        Parameters
        ----------
        optimizer: string
            optimizer to be used (NS or MC)
        """

        if self.single_parameter_fits:
            self.single_parameter_analysis(optimizer)
        else:
            self.global_analysis(optimizer)

    def chi2_scan(self, n_replica, compute_bounds):
        r"""Run an individual :math:`\chi^2` scan.

        Parameters
        ----------
        n_replica: int
            number of replicas to use.
            If 0 only the :math:`\chi^2` experiemental data
            will be computed.
        compute_bounds: bool
            if True compute and save the :math:`\chi^2` bounds.
        """
        scan = Scanner(self.run_card, n_replica)
        if compute_bounds:
            scan.compute_bounds()
        scan.compute_scan()
        scan.plot_scan()
