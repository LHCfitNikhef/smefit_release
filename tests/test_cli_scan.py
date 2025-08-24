# -*- coding: utf-8 -*-
import json
import pathlib
from typing import Any

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


@pytest.mark.parametrize(
    "runcard_filename,result_id",
    [
        ("scan_wc.yaml", "scan_wc"),
        ("scan_mass.yaml", "scan_mass"),
        ("scan_mass_with_rge.yaml", "scan_mass_with_rge"),
    ],
    ids=["scan_wc", "scan_mass", "scan_mass_with_rge"],
)
def test_cli_scan_matches_precomputed(
    tmp_path: pathlib.Path, runcard_filename: str, result_id: str
):
    base_dir = pathlib.Path(__file__).parent / "fit_tests"
    runcard_src = base_dir / runcard_filename
    exp_scan_file = base_dir / "test_results" / result_id / "chi2_scan.json"

    assert runcard_src.is_file(), "Expected scan runcard not found"
    assert exp_scan_file.is_file(), "Expected precomputed scan not found"

    # Prepare runcard with absolute paths and temporary result path
    rc = _load_yaml(runcard_src)
    rc["data_path"] = str((base_dir / "test_commondata").resolve())
    rc["theory_path"] = str((base_dir / "test_theory").resolve())
    rc["result_path"] = str(tmp_path / "scan_results")
    rc.setdefault("result_ID", result_id)

    runcard_dst = tmp_path / runcard_src.name
    _dump_yaml(runcard_dst, rc)

    # Run CLI: use 5 scan points to match fixtures
    runner = CliRunner()
    result = runner.invoke(
        base_command, ["SCAN", "-s", "5", str(runcard_dst)], catch_exceptions=False
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    produced_scan = tmp_path / "scan_results" / result_id / "chi2_scan.json"
    assert produced_scan.is_file(), "chi2_scan.json was not produced"

    with open(produced_scan, encoding="utf-8") as f:
        got = json.load(f)
    with open(exp_scan_file, encoding="utf-8") as f:
        exp = json.load(f)

    # Same set of scanned parameters
    assert set(got.keys()) == set(exp.keys())

    # Compare per-parameter scan content
    for name, exp_entry in exp.items():
        assert name in got, f"Missing scan for {name}"
        got_entry = got[name]

        # x grid: should match up to tight tolerance
        np.testing.assert_allclose(
            np.asarray(got_entry["x"], float),
            np.asarray(exp_entry["x"], float),
            rtol=1e-9,
            atol=1e-12,
        )

        # n_datapoints exact match
        assert int(got_entry["n_datapoints"]) == int(exp_entry["n_datapoints"])

        # replica 0 chi2 values (deterministic run): allow tiny numeric tolerance
        np.testing.assert_allclose(
            np.asarray(got_entry["0"], float),
            np.asarray(exp_entry["0"], float),
            rtol=1e-6,
            atol=1e-8,
        )
