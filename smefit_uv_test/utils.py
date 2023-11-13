import arviz
import logging
import pathlib
import rich
import json
import pandas as pd

import matplotlib.pyplot as plt
from rich.logging import RichHandler
from smefit.fit_manager import FitManager
from smefit.analyze.coefficients_utils import compute_confidence_level

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

here = pathlib.Path("/data/theorie/jthoeve/smefit_release/smefit_uv_test")
parent = pathlib.Path("/data/theorie/jthoeve/smefit_release")

# here = pathlib.Path(__file__).absolute().parents[0]
# parent = pathlib.Path(__file__).absolute().parents[1]

def load_results(fit):
    fit.load_results()
    return fit.results


def compute_chi2(fit):
    fit.compute_chi2()
    return fit.chi2_rep


def build_inv_posterior(results, inv_list):
    for i, inv in enumerate(inv_list):
        results[f"inv{i+1}"] = inv(results)
    return results


def inspect_model(MODEL_SPECS, build_uv_posterior, inv_list, check_constrain=None):

    colllection = MODEL_SPECS["collection"]
    model_id = MODEL_SPECS["id"]
    pto = MODEL_SPECS["pto"]
    eft = MODEL_SPECS["eft"]

    fit_name_uv = f"{colllection}_UV_{model_id}_{pto}_{eft}_NS"

    # load results and build posterior for uv couplings
    fit_uv = FitManager(parent / "results", fit_name_uv)
    results_uv = load_results(fit_uv)

    # construct the invariants
    results_uv = build_inv_posterior(results_uv, inv_list)

    results_uv_dict = results_uv.to_dict(orient='list')
    with open(parent / "results" / fit_name_uv / "inv_posterior.json", 'w', encoding='utf8') as f:
        json.dump(results_uv_dict, f)


def plot_cl_table(results_uv):

    df = pd.DataFrame(
        index=results_uv.columns,
        columns=[
            "UV -",
            "WC -",
            "UV +",
            "WC +",
            "UV mid",
            "WC mid",
        ],
    )
    for op in results_uv:
        uv_left, uv_right = arviz.hdi(results_uv[op].values).tolist()
        df.loc[op, "UV -"] = uv_left
        df.loc[op, "UV mid"] = results_uv[op].mean()
        df.loc[op, "UV +"] = uv_right
    df["percent diff"] = (df["UV mid"] - df["WC mid"]) / df["UV mid"] * 100
    rich.print(df)