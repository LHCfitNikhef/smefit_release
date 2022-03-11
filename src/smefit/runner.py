# -*- coding: utf-8 -*-
import pathlib
import subprocess
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
        runcard_folder : pathlib.Path
            root path
    """

    def __init__(self, runcard_folder, run_card_name):

        self.runcard_folder = pathlib.Path(runcard_folder)
        self.run_card_name = run_card_name

        print(20 * "  ", r" ____  __  __ _____ _____ _ _____ ")
        print(20 * "  ", r"/ ___||  \/  | ____|  ___(_)_   _|")
        print(20 * "  ", r"\___ \| |\/| |  _| | |_  | | | |  ")
        print(20 * "  ", r" ___) | |  | | |___|  _| | | | |  ")
        print(20 * "  ", r"|____/|_|  |_|_____|_|   |_| |_|  ")
        print()
        print(18 * "  ", "A Standard Model Effective Field Theory Fitter")

    def load_config(self):
        """
        Load runcard file
        """
        config = {}
        with open(
            self.runcard_folder / f"{self.run_card_name}.yaml", encoding="utf-8"
        ) as f:
            config = yaml.safe_load(f)

        return config

    def setup_result_folder(self, result_folder):
        """
        Read yaml card, update the configuration paths
        and build the folder directory.

        Parameters
        ----------
            filename : str
                fit card name

        """
        # Construct results folder
        result_folder = pathlib.Path(result_folder)
        res_folder_fit = result_folder / self.run_card_name

        subprocess.call(f"mkdir -p {result_folder}", shell=True)
        if res_folder_fit.exist():
            raise Warning(f"{res_folder_fit} already found, overwriting old results")
        subprocess.call(f"mkdir -p {res_folder_fit}", shell=True)

        # Copy yaml runcard to results folder
        copyfile(
            self.runcard_folder / f"{self.run_card_name}.yaml",
            res_folder_fit / f"{self.run_card_name}.yaml",
        )

    def ns(self, input_card):
        """
        Run a fit with |NS| given the fit name

        Parameters
        ----------
            input_card : str, dict
                fit card name
        """
        print("RUNNING: Nested Sampling Fit ")

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            if isinstance(input_card, dict):
                config = input_card
            else:
                config = self.load_config()
            self.setup_result_folder(config["result_path"])
        else:
            config = None

        config = comm.bcast(config, root=0)

        # Run optimizer
        opt = NSOptimizer.from_dict(config)
        opt.run_sampling()
