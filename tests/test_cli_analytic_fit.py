# -*- coding: utf-8 -*-
"""CLI integration test for the Analytic (A) fit.

This test runs `smefit A <runcard>` on a small fake setup and
compares deterministic outputs (best-fit point, max log-likelihood,
and free parameter names) against precomputed results.
"""
from __future__ import annotations

import json
import pathlib
import pickle
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
    "runcard_filename,result_id,expect_rge",
    [
        ("analytic_fit_glob.yaml", "analytic_fit_glob", False),
        ("analytic_fit_glob_with_rge.yaml", "analytic_fit_glob_with_rge", True),
    ],
    ids=["analytic_glob", "analytic_glob_rge"],
)
def test_cli_analytic_fit_matches_precomputed(
    tmp_path: pathlib.Path, runcard_filename: str, result_id: str, expect_rge: bool
):
    # Paths to fixtures and runcard
    base_dir = pathlib.Path(__file__).parent / "fit_tests"
    runcard_src = base_dir / runcard_filename
    precomputed_dir = base_dir / "test_results" / result_id
    precomputed_file = precomputed_dir / "fit_results.json"
    rge_matrix_file = precomputed_dir / "rge_matrix.pkl"

    assert runcard_src.is_file(), "Expected test runcard not found"
    assert precomputed_file.is_file(), "Expected precomputed results not found"
    if expect_rge:
        assert rge_matrix_file.is_file(), "Expected RGE matrix not found"

    # Prepare a copy of the runcard with absolute paths and a temp results folder.
    rc = _load_yaml(runcard_src)
    rc["data_path"] = str((base_dir / "test_commondata").resolve())
    rc["theory_path"] = str((base_dir / "test_theory").resolve())
    rc["result_path"] = str(tmp_path / "fit_results")
    rc.setdefault("result_ID", result_id)

    # Write the adjusted runcard
    runcard_dst = tmp_path / runcard_src.name
    _dump_yaml(runcard_dst, rc)

    # Run the CLI: smefit A <runcard>
    runner = CliRunner()
    result = runner.invoke(
        base_command, ["A", str(runcard_dst)], catch_exceptions=False
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    # Load our produced results
    produced_fit = tmp_path / "fit_results" / result_id / "fit_results.json"
    assert produced_fit.is_file(), "fit_results.json was not produced"

    with open(produced_fit, encoding="utf-8") as f:
        got = json.load(f)
    with open(precomputed_file, encoding="utf-8") as f:
        exp = json.load(f)

    # Optional: compare RGE matrix when expected
    if expect_rge:
        produced_rge = tmp_path / "fit_results" / result_id / "rge_matrix.pkl"
        assert produced_rge.is_file(), "rge_matrix.pkl was not produced"
        with open(rge_matrix_file, "rb") as f:
            rge_matrix_precomp = pickle.load(f)
        with open(produced_rge, "rb") as f:
            rge_matrix_produced = pickle.load(f)

        assert isinstance(rge_matrix_produced, list)
        assert isinstance(rge_matrix_precomp, list)
        assert len(rge_matrix_produced) == len(rge_matrix_precomp)
        for got_df, exp_df in zip(rge_matrix_produced, rge_matrix_precomp):
            assert list(got_df.index) == list(exp_df.index)
            assert list(got_df.columns) == list(exp_df.columns)
            np.testing.assert_allclose(got_df.values, exp_df.values, rtol=1e-4)

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
            rtol=1e-4,
        )

    # - max log-likelihood and evidence (logZ)
    np.testing.assert_allclose(
        got["max_loglikelihood"], exp["max_loglikelihood"], rtol=1e-4
    )
    np.testing.assert_allclose(got["logz"], exp["logz"], rtol=1e-4)

    # compare mean and standard deviation of posterior samples (looser tolerance)
    for name in exp["samples"].keys():
        assert name in got["samples"], f"Missing samples for {name}"
        np.testing.assert_allclose(
            np.mean(got["samples"][name]), np.mean(exp["samples"][name]), rtol=1e-1
        )
        np.testing.assert_allclose(
            np.std(got["samples"][name]), np.std(exp["samples"][name]), rtol=1e-1
        )


def test_cli_analytic_fit_indiv_matches_precomputed(tmp_path: pathlib.Path):
    # Paths to fixtures and runcard
    base_dir = pathlib.Path(__file__).parent / "fit_tests"
    runcard_src = base_dir / "analytic_fit_indiv.yaml"
    precomputed_file = (
        base_dir / "test_results" / "analytic_fit_indiv" / "fit_results.json"
    )

    assert runcard_src.is_file(), "Expected test runcard not found"
    assert precomputed_file.is_file(), "Expected precomputed results not found"

    # Prepare a copy of the runcard with absolute paths and a temp results folder.
    rc = _load_yaml(runcard_src)
    rc["data_path"] = str((base_dir / "test_commondata").resolve())
    rc["theory_path"] = str((base_dir / "test_theory").resolve())
    rc["result_path"] = str(tmp_path / "fit_results")
    result_id = rc.get("result_ID", "analytic_fit_indiv")

    runcard_dst = tmp_path / "analytic_fit_indiv.yaml"
    _dump_yaml(runcard_dst, rc)

    # Run the CLI: smefit A <runcard>
    runner = CliRunner()
    result = runner.invoke(
        base_command, ["A", str(runcard_dst)], catch_exceptions=False
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    # Load produced and expected results
    produced = tmp_path / "fit_results" / result_id / "fit_results.json"
    assert produced.is_file(), "fit_results.json was not produced"
    with open(produced, encoding="utf-8") as f:
        got = json.load(f)
    with open(precomputed_file, encoding="utf-8") as f:
        exp = json.load(f)

    # Basic flags and parameter names
    assert got.get("single_parameter_fits", True) is True
    assert set(got["free_parameters"]) == set(
        exp["free_parameters"]
    )  # order-insensitive

    # Per-operator deterministic fields: best_fit_point, max_loglikelihood, logz are dicts
    for name in exp["best_fit_point"].keys():
        assert name in got["best_fit_point"], f"Missing best-fit for {name}"
        np.testing.assert_allclose(
            got["best_fit_point"][name], exp["best_fit_point"][name], rtol=1e-4
        )

    for name in exp["max_loglikelihood"].keys():
        assert name in got["max_loglikelihood"], f"Missing max_loglikelihood for {name}"
        np.testing.assert_allclose(
            got["max_loglikelihood"][name], exp["max_loglikelihood"][name], rtol=1e-4
        )

    for name in exp["logz"].keys():
        assert name in got["logz"], f"Missing logz for {name}"
        np.testing.assert_allclose(got["logz"][name], exp["logz"][name], rtol=1e-4)

    # Samples: compare summary stats per operator with loose tolerance
    for name in exp["samples"].keys():
        assert name in got["samples"], f"Missing samples for {name}"
        np.testing.assert_allclose(
            np.mean(got["samples"][name]), np.mean(exp["samples"][name]), rtol=1e-1
        )
        np.testing.assert_allclose(
            np.std(got["samples"][name]), np.std(exp["samples"][name]), rtol=1e-1
        )
