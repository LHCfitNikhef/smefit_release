# -*- coding: utf-8 -*-
import pathlib
import subprocess
import warnings
from shutil import copyfile

import yaml
from mpi4py import MPI

from .optimize.ns import NSOptimizer


class Runner:
    """
    Container for all the possible |SMEFiT| run methods.

    Init the root path of the package where tables,
    results, plot config and reports are be stored

    Parameters
    ----------
        run_card : dict
            run card dictionary
        runcard_folder: pathlib.Path, None
            path to runcard folder if already present
    """

    def __init__(self, run_card, runcard_folder=None):

        print(20 * "  ", r" ____  __  __ _____ _____ _ _____ ")
        print(20 * "  ", r"/ ___||  \/  | ____|  ___(_)_   _|")
        print(20 * "  ", r"\___ \| |\/| |  _| | |_  | | | |  ")
        print(20 * "  ", r" ___) | |  | | |___|  _| | | | |  ")
        print(20 * "  ", r"|____/|_|  |_|_____|_|   |_| |_|  ")
        print()
        print(18 * "  ", "A Standard Model Effective Field Theory Fitter")

        self.run_card = run_card
        self.runcard_folder = runcard_folder
        self.setup_result_folder()

    def setup_result_folder(self):
        """
        Create result folder and copy the runcard there
        """
        # Construct results folder
        run_card_id = self.run_card["result_ID"]
        result_folder = pathlib.Path(self.run_card["result_path"])
        res_folder_fit = result_folder / run_card_id

        subprocess.call(f"mkdir -p {result_folder}", shell=True)
        if res_folder_fit.exists():
            warnings.warn(f"{res_folder_fit} already found, overwriting old results")
        subprocess.call(f"mkdir -p {res_folder_fit}", shell=True)

        # Copy yaml runcard to results folder or dump it
        # in case no given file is passed
        runcard_copy = res_folder_fit / f"{run_card_id}.yaml"
        if self.runcard_folder is None:
            with open(runcard_copy, encoding="utf-8") as f:
                yaml.dump(self.run_card, f, default_flow_style=False)
        else:
            copyfile(
                self.runcard_folder / f"{run_card_id}.yaml",
                runcard_copy,
            )

    @classmethod
    def from_file(cls, runcard_folder, run_card_name):
        """
        Create Runner from a runcard file

        Parameters
        ----------
        runcard_folder: pathlib.Path, str
            path to runcard folder if already present
        run_card_name : srt
            run card name

        Returns
        -------
            runner: `smefit.runner.Runner`
                instance of class Runner
        """
        config = {}
        # load file
        runcard_folder = pathlib.Path(runcard_folder)
        with open(runcard_folder / f"{run_card_name}.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # set result ID to runcard name by default
        if "result_ID" not in config:
            config["result_ID"] = run_card_name

        return cls(config, runcard_folder)

    def ns(self):
        """
        Run a fit with |NS|
        """
        print("RUNNING: Nested Sampling Fit ")

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            config = self.run_card
            opt = NSOptimizer.from_dict(config)
        else:
            opt = None

        # Run optimizer
        opt = comm.bcast(opt, root=0)
        opt.run_sampling()
