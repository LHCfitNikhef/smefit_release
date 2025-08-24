# -*- coding: utf-8 -*-
import pathlib
import shutil
from typing import Any, Optional

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from smefit.cli import base_command


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(path: pathlib.Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _as_array(x):
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=float)
    # scalar
    return np.asarray([x], dtype=float)


@pytest.mark.parametrize(
    "proj_card,exp_rel_dir,noise,seed",
    [
        ("proj_SM.yaml", "SM_proj", "L0", None),
        ("proj_BSM.yaml", "BSM_proj", "L0", None),
        ("proj_SM.yaml", "SM_proj/seed_1", "L1", 1),
        ("proj_BSM.yaml", "BSM_proj/seed_1", "L1", 1),
    ],
    ids=["PROJ_SM_L0", "PROJ_BSM_L0", "PROJ_SM_L1_seed1", "PROJ_BSM_L1_seed1"],
)
def test_cli_proj_matches_fixtures(
    tmp_path: pathlib.Path,
    proj_card: str,
    exp_rel_dir: str,
    noise: str,
    seed: Optional[int],
):
    base_dir = pathlib.Path(__file__).parent / "fit_tests"
    card_src = base_dir / proj_card
    exp_dir = base_dir / "test_results" / exp_rel_dir

    assert card_src.is_file(), "Projection runcard not found"
    assert exp_dir.is_dir(), "Expected results folder not found"

    # Prepare temporary theory folder to avoid mutating repo fixtures
    theory_src = base_dir / "test_theory"
    theory_tmp = tmp_path / "theory"
    shutil.copytree(theory_src, theory_tmp)

    # Prepare runcard with absolute paths and temporary projection output path
    card = _load_yaml(card_src)
    card["commondata_path"] = str((base_dir / "test_commondata").resolve())
    card["theory_path"] = str(theory_tmp)
    out_root = tmp_path / "projections" / exp_rel_dir.split("/")[0]
    card["projections_path"] = str(out_root)

    card_dst = tmp_path / card_src.name
    _dump_yaml(card_dst, card)

    # Run CLI: use lumi=3000 to match fixtures; set noise/seed per case
    runner = CliRunner()
    args = ["PROJ", "--lumi", "3000", "--noise", noise]
    if seed is not None:
        args += ["--seed", str(seed)]
    args += [str(card_dst)]
    result = runner.invoke(base_command, args, catch_exceptions=False)
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    # BSM projection produces an RGE matrix artifact
    if exp_rel_dir.startswith("BSM_proj"):
        assert (out_root / "rge_matrix.pkl").is_file(), "Missing RGE matrix in output"

    # Compare each produced YAML with expected, allowing small numeric tolerances
    atol = 1e-6
    rtol = 1e-4

    got_dir = out_root if seed is None else (out_root / f"seed_{seed}")

    for exp_file in sorted(exp_dir.glob("*_proj.yaml")):
        got_file = got_dir / exp_file.name
        assert got_file.is_file(), f"Missing projection output: {got_file.name}"

        exp = _load_yaml(exp_file)
        got = _load_yaml(got_file)

        # Basic metadata checks
        assert got["dataset_name"] == exp["dataset_name"]
        assert float(got["luminosity"]) == pytest.approx(float(exp["luminosity"]))
        assert int(got["num_data"]) == int(exp["num_data"])
        assert int(got["num_sys"]) == int(exp["num_sys"])

        # Numeric arrays
        np.testing.assert_allclose(
            _as_array(got["data_central"]),
            _as_array(exp["data_central"]),
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            _as_array(got["statistical_error"]),
            _as_array(exp["statistical_error"]),
            rtol=rtol,
            atol=atol,
        )

        # Systematics: stored as list-of-lists even for num_data=1 after our normalization
        got_sys = _as_array(got["systematics"]).reshape(-1)
        exp_sys = _as_array(exp["systematics"]).reshape(-1)
        np.testing.assert_allclose(got_sys, exp_sys, rtol=rtol, atol=atol)

        # Names and types
        # sys_names may be a list or single str in fixtures depending on num_sys
        def _as_list(v):
            if v is None:
                return []
            return v if isinstance(v, list) else [v]

        assert _as_list(got.get("sys_names")) == _as_list(exp.get("sys_names"))
        assert _as_list(got.get("sys_type")) == _as_list(exp.get("sys_type"))
