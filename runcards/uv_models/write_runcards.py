# -*- coding: utf-8 -*-
import argparse
import copy
import pathlib

import yaml

here = pathlib.Path(__file__).parent

MAX_VALUE = 1000


def load_base() -> dict:
    """Load basic runcard."""
    with open(here / "base_runcard.yaml", encoding="utf-8") as f:
        card = yaml.safe_load(f)
    return card


def parse_UV_coeffs(model_dict: dict) -> dict:
    """Parse the UV coefficient dictionay."""
    coeff_dict = {}
    free_dofs = []

    # add uv couplings
    for c in model_dict["UV couplings"]:
        free_dofs.append(c)
        # TODO: do we have sign restricitions here ??
        coeff_dict[c] = {"min": -MAX_VALUE, "max": MAX_VALUE}

    # now add the non linear relations
    for coeff, rel in model_dict.items():
        if not coeff.startswith("c"):
            continue

        if len(rel) == 1:
            # drop coeff fixed to a value
            value = rel[0][free_dofs[0]][0]
            if value != 0:
                coeff_dict[f"O{coeff[1:]}"] = {"constrain": True, "value": value}
                continue
            continue

        new_constrain = []
        for sum_element in rel:
            new_addend = {}
            is_non_zero = False
            for c, pair in sum_element.items():
                # if there is 0 drop the constrain
                if pair[0] == 0:
                    is_non_zero = True
                    continue
                if pair == [1, 0]:
                    continue
                if not is_non_zero:
                    new_addend[c] = pair
            if new_addend != {}:
                new_constrain.append(new_addend)

        coeff_dict[f"O{coeff[1:]}"] = {
            "constrain": new_constrain,
            "min": -MAX_VALUE,
            "max": MAX_VALUE,
        }
    return coeff_dict


def parse_WC_coeffs(model_dict: dict) -> dict:
    coeff_dict = {}
    # add uv couplings
    for coeff, spec_dict in model_dict.items():
        if coeff.startswith("c"):
            # remove coeffs which are 0
            if "constrain" in spec_dict and spec_dict["constrain"]:
                if "value" in spec_dict and spec_dict["value"] == 0:
                    continue
            if "min" not in spec_dict:
                spec_dict["min"] = -MAX_VALUE
            if "max" not in spec_dict:
                spec_dict["max"] = MAX_VALUE
            coeff_dict[f"O{coeff[1:]}"] = spec_dict
    return coeff_dict


def dump_runcard(
    collection: str,
    idx_model: int,
    eft_order: str,
    pto: str,
    fitting_mode: str,
    is_uv: bool,
) -> None:
    """Parse a model card to a SMEFiT runcard."""
    if is_uv:
        file_path = f"UV_scan/{collection}/out_UV_dict_Coll_{collection}_Mod_{idx_model}_Mass_1_Tree.yaml"
    else:
        file_path = f"WC_scan/{collection}/dict_WC_scan_Coll_{collection}_Mod_{idx_model}_Mass_1_Tree.yaml"

    with open(here / file_path, "r") as f:
        model_dict = yaml.safe_load(f)

    runcard = load_base()

    # orders
    runcard["order"] = pto
    runcard["use_quad"] = eft_order == "HO"

    # names
    runcard["resultID"] = f"Model_{is_uv}_{idx_model}_{pto}_{eft_order}"
    runcard["Model name"] = model_dict["Model name"]
    runcard["UV Collection"] = model_dict["UV Collection"]
    runcard["UV model"] = model_dict["UV model"]
    runcard["uv_couplings"] = is_uv

    if is_uv:
        coeff_dict = parse_UV_coeffs(model_dict)
        flag = "UV"
    else:
        coeff_dict = parse_WC_coeffs(model_dict)
        flag = "WC"
    runcard["coefficients"] = coeff_dict
    with open(
        f"{here.parent}/{collection}_{flag}_{idx_model}_{pto}_{eft_order}_{fitting_mode}.yaml",
        "w",
    ) as f:
        yaml.dump(runcard, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="SMEFiT_runcards",
        description="Write SMEFIT runcards for WC/UV models",
    )
    parser.add_argument("-i", "--model_idx", help="Model ID", type=int, required=True)
    parser.add_argument(
        "-e", "--eft_order", help="EFT order: HO, NHO", type=str, required=True
    )
    parser.add_argument(
        "-o", "--qcd_order", help="QCD order: LO, NLO", type=str, required=True
    )
    parser.add_argument(
        "-m", "--mode", help="Fitting mode: NS, MC", type=str, default="NS"
    )
    parser.add_argument(
        "-c",
        "--collection",
        help="Model collection: Granada, Fitmaker",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-u",
        "--is_uv",
        help="True for UV model",
        action="store_true",
    )
    args = parser.parse_args()

    dump_runcard(
        args.collection,
        args.model_idx,
        args.eft_order,
        args.qcd_order,
        args.mode,
        args.is_uv,
    )
