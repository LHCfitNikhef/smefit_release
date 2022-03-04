# -*- coding: utf-8 -*-
import subprocess
from shutil import copyfile

import yaml

from .optimize_ns import NSOptimizer
from .utils import set_paths


class Runner:
    """
    Container for all the possible |SMEFiT| run methods.

    Init the root path of the package where tables,
    results, plot config and reports are be stored

    Parameters
    ----------
        root_path : pathlib.Path
            root path
    """

    def __init__(self, root_path):

        self.root_path = root_path

        print(20 * "  ", r" ____  __  __ _____ _____ _ _____ ")
        print(20 * "  ", r"/ ___||  \/  | ____|  ___(_)_   _|")
        print(20 * "  ", r"\___ \| |\/| |  _| | |_  | | | |  ")
        print(20 * "  ", r" ___) | |  | | |___|  _| | | | |  ")
        print(20 * "  ", r"|____/|_|  |_|_____|_|   |_| |_|  ")
        print()
        print(18 * "  ", "A Standard Model Effective Field Theory Fitter")

    def setup_config(self, filename):
        """
        Read yaml card, update the configuration paths
        and build the folder directory.

        Parameters
        ----------
            filename : str
                fit card name
        Returns
        -------
            config: dict
                configuration dict
        """
        config = {}
        with open(f"{self.root_path}/runcards/{filename}.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        config.update(set_paths(self.root_path, config["order"], config["resultID"]))

        # Construct results folder
        res_folder = f"{self.root_path}/results"
        res_folder_fit = config["results_path"]

        subprocess.call(f"mkdir -p {res_folder}", shell=True)
        subprocess.call(f"mkdir -p {res_folder_fit}", shell=True)

        # Copy yaml runcard to results folder
        copyfile(
            f"{self.root_path}/runcards/{filename}.yaml",
            f"{config['results_path']}/{filename}.yaml",
        )

        return config

    def ns(self, input_card):
        """
        Run a fit with |NS| given the fit name

        Parameters
        ----------
            input_card : str
                fit card name
        """
        config = self.setup_config(input_card)
        opt = NSOptimizer(config)
        opt.run_sampling()
