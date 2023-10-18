import yaml
import pathlib
import numpy as np

from ..loader import Loader, load_datasets
from ..log import logging
from ..compute_theory import make_predictions

_logger = logging.getLogger(__name__)


class Projection:
    def __init__(
        self,
        commondata_path,
        theory_path,
        dataset_names,
        projections_path,
        coefficients,
        order,
        use_quad,
        rot_to_fit_basis,
    ):
        self.commondata_path = commondata_path
        self.theory_path = theory_path
        self.dataset_names = dataset_names
        self.projections_path = projections_path
        self.coefficients = coefficients
        self.order = order
        self.use_quad = use_quad
        self.rot_to_fit_basis = rot_to_fit_basis

    @classmethod
    def from_config(cls, projection_card):
        with open(projection_card, encoding="utf-8") as f:
            projection_config = yaml.safe_load(f)

        commondata_path = pathlib.Path(projection_config["commondata_path"]).absolute()
        theory_path = pathlib.Path(projection_config["theory_path"]).absolute()
        projections_path = pathlib.Path(
            projection_config["projections_path"]
        ).absolute()
        dataset_names = projection_config["datasets"]

        coefficients = projection_config.get("coefficients", [])
        order = projection_config.get("order", "LO")
        use_quad = projection_config.get("use_quad", False)
        rot_to_fit_basis = projection_config.get("rot_to_fit_basis", None)

        return cls(
            commondata_path,
            theory_path,
            dataset_names,
            projections_path,
            coefficients,
            order,
            use_quad,
            rot_to_fit_basis,
        )

    def compute_cv_projection(self, dataset_name):
        _logger.info(f"Building projection for : {dataset_name}")
        dataset = load_datasets(
            self.commondata_path,
            dataset_name,
            self.coefficients,
            self.order,
            self.use_quad,
            False,
            False,
            False,
            theory_path=self.theory_path,
        )

        cv = dataset.SMTheory
        if self.coefficients:
            _logger.warning(
                f"Some coefficients are specified in the runcard: EFT correction will be used for the central values"
            )
            coefficient_values = []
            for coeff in dataset.OperatorsNames:
                coefficient_values.append(self.coefficients[coeff]["value"])

            cv = make_predictions(dataset, coefficient_values, self.use_quad, False)
        return cv

    def build_projection(self, reduction_factor):
        for dataset in self.dataset_names:
            # load original experimental set

            path_to_dataset = self.commondata_path / f"{dataset}.yaml"

            with open(path_to_dataset, encoding="utf-8") as f:
                data_dict = yaml.safe_load(f)

            # get new cv
            cv = self.compute_cv_projection(dataset)

            # use sm predictions for central values
            data_dict["data_central"] = cv.tolist()
            # replace stat with the new one
            stat = np.asarray(data_dict["statistical_error"])
            data_dict["statistical_error"] = (reduction_factor * stat).tolist()

            projection_folder = self.projections_path
            projection_folder.mkdir(exist_ok=True)
            with open(f"{projection_folder}/{dataset}_projection.yaml", "w") as file:
                yaml.dump(data_dict, file, sort_keys=False)
