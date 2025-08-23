# -*- coding: utf-8 -*-
"""CLI integration test for the Analytic (A) fit.

This test runs `smefit A <runcard>` on a small fake setup and
compares deterministic outputs (best-fit point, max log-likelihood,
and free parameter names) against precomputed results.
"""
from __future__ import annotations

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


@pytest.mark.integration
def test_cli_analytic_fit_matches_precomputed(tmp_path: pathlib.Path):
    # Paths to fixtures and runcard
    base_dir = pathlib.Path(__file__).parent
    runcard_src = base_dir / "analytic_fit_glob.yaml"
    precomputed_file = (
        base_dir / "test_results" / "analytic_fit_glob" / "fit_results.json"
    )

    assert runcard_src.is_file(), "Expected test runcard not found"
    assert precomputed_file.is_file(), "Expected precomputed results not found"

    # Prepare a copy of the runcard with absolute paths and a temp results folder.
    rc = _load_yaml(runcard_src)

    rc["data_path"] = str((base_dir / "test_commondata").resolve())
    rc["theory_path"] = str((base_dir / "test_theory").resolve())
    rc["result_path"] = str(tmp_path / "fit_results")
    # Keep the same result_ID so we can also compare with the precomputed folder name
    result_id = rc.get("result_ID", "analytic_fit_glob")

    # Write the adjusted runcard
    runcard_dst = tmp_path / "analytic_fit_glob.yaml"
    _dump_yaml(runcard_dst, rc)

    # Run the CLI: smefit A <runcard>
    runner = CliRunner()
    result = runner.invoke(
        base_command, ["A", str(runcard_dst)], catch_exceptions=False
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    # Load our produced results
    produced = tmp_path / "fit_results" / result_id / "fit_results.json"
    assert produced.is_file(), "fit_results.json was not produced"

    with open(produced, encoding="utf-8") as f:
        got = json.load(f)
    with open(precomputed_file, encoding="utf-8") as f:
        exp = json.load(f)

    # Compare deterministic content
    # - free parameter names (set equality, order-insensitive)
    assert set(got["free_parameters"]) == set(
        exp["free_parameters"]
    ), "Free parameters mismatch"

    # - best-fit point per parameter
    for name in exp["best_fit_point"].keys():
        assert name in got["best_fit_point"], f"Missing best-fit for {name}"
        np.testing.assert_allclose(
            got["best_fit_point"][name],
            exp["best_fit_point"][name],
            rtol=1e-9,
            atol=1e-12,
        )

    # - max log-likelihood should match within tolerance
    np.testing.assert_allclose(
        got["max_loglikelihood"], exp["max_loglikelihood"], rtol=1e-9, atol=1e-12
    )

    # - evidence (logZ) is deterministic for the analytic solution; compare loosely
    np.testing.assert_allclose(got["logz"], exp["logz"], rtol=1e-7, atol=1e-10)

    # compare mean and standard deviation of posterior samples
    # They should match within a reasonable tolerance, set to 10%
    # since it depends on the samples that experience fluctuations
    for name in exp["samples"].keys():
        assert name in got["samples"], f"Missing samples for {name}"
        np.testing.assert_allclose(
            np.mean(got["samples"][name]),
            np.mean(exp["samples"][name]),
            rtol=1e-1,
        )
        np.testing.assert_allclose(
            np.std(got["samples"][name]),
            np.std(exp["samples"][name]),
            rtol=1e-1,
        )
