import yaml


class RUNNER:
    """
    Class containing all the possible smefit run methods.

    Init the root path of the package where tables,
    results, plot config and reports are be stored

    Parameters
    ----------
        path : str
            root path
    """

    def __init__(self, path):

        self.root_path = path

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

        import subprocess
        import yaml
        from shutil import copyfile
        from .utils import set_paths

        config = {}
        with open(f"{self.root_path}/runcards/{filename}.yaml") as f:
            config = yaml.safe_load(f)

        config.update(set_paths(self.root_path, config["order"], config["resultID"]))

        # Construct results folder
        res_folder = f"{self.root_path}/results"
        res_folder_run = config["results_path"]

        subprocess.call(f"mkdir -p {res_folder}", shell=True)
        subprocess.call(f"mkdir -p {res_folder_run}", shell=True)

        # Copy yaml runcard to results folder
        copyfile(
            f"{self.root_path}/runcards/{filename}.yaml",
            f"{config['results_path']}/{filename}.yaml",
        )

        return config

    def ns(self, input_card):
        """
        Run a fit with Nested Sampling given the fit name

        Parameters
        ----------
            input_card : str
                fit card name
        """

        config = self.setup_config(input_card)
        from .optimize_ns import OPTIMIZE

        opt = OPTIMIZE(config)
        opt.run_sampling()

        return 0
