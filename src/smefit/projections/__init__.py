import yaml
import pathlib

from ..loader import Loader
from ..log import logging

_logger = logging.getLogger(__name__)


class Projection:
    def __init__(
        self,
        commondata_path,
        theory_path,
        dataset_name,
        projections_path,
        coefficients,
        order,
        use_quad,
        rot_to_fit_basis,
    ):
        self.commondata_path = commondata_path
        self.theory_path = theory_path
        self.dataset_name = dataset_name
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
        dataset_name = projection_config["dataset"]

        if projection_config["use_smeft"]:
            coefficients = projection_config["coefficients"]
            order = projection_config["order"]
            use_quad = projection_config["use_quad"]
            rot_to_fit_basis = projection_config["rot_to_fit_basis"]
        else:
            coefficients = []
            order = "LO"
            use_quad = False
            rot_to_fit_basis = None

        return cls(
            commondata_path,
            theory_path,
            dataset_name,
            projections_path,
            coefficients,
            order,
            use_quad,
            rot_to_fit_basis,
        )

    def load_dataset(self):
        Loader.commondata_path = self.commondata_path
        Loader.theory_path = self.theory_path

        dataset = Loader(
            self.dataset_name,
            self.coefficients,
            self.order,
            self.use_quad,
            False,
            False,
            self.rot_to_fit_basis,
        )
        import pdb; pdb.set_trace()
        return dataset.sm_prediction, dataset.stat_error

    def build_projection(self, reduction_factor):
        # load original experimental set
        _logger.info(f"Reading experimental dataset : {self.dataset_name}")
        path_to_dataset = self.commondata_path / f"{self.dataset_name}.yaml"

        with open(path_to_dataset, encoding="utf-8") as f:
            data_dict = yaml.safe_load(f)

        # get sm predictions and stat
        sm, stat = self.load_dataset()

        # use sm predictions for central values
        data_dict["data_central"] = sm
        # replace stat with the new one
        data_dict["statistical_error"] = (reduction_factor * stat).tolist()

        projection_folder = self.projections_path
        projection_folder.mkdir(exist_ok=True)
        with open(
            f"{projection_folder}/{self.dataset_name}_projection.yaml", "w"
        ) as file:
            yaml.dump(data_dict, file, sort_keys=False)
