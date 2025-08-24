# -*- coding: utf-8 -*-
"""CLI integration test for the ultranest (NS) fit.

This test runs `smefit NS <runcard>` on a small fake setup and
compares deterministic outputs (best-fit point, max log-likelihood,
and free parameter names) against precomputed results.
"""
from __future__ import annotations

import json
import pathlib
import pickle
from typing import Any

import numpy as np
import yaml
from click.testing import CliRunner

from smefit.cli import base_command


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(path: pathlib.Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _quantiles(a: list[float] | np.ndarray, qs=(0.025, 0.975)) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    return np.quantile(arr, qs)


def test_cli_ultranest_lin_fit_matches_precomputed(tmp_path: pathlib.Path):
    # Paths to fixtures and runcard
    # The fixtures (runcard, data, theory, expected results) live under tests/fit_tests
    base_dir = pathlib.Path(__file__).parent / "fit_tests"
    runcard_src = base_dir / "ultranest_fit_lin_glob.yaml"
    precomputed_file = (
        base_dir / "test_results" / "ultranest_fit_lin_glob" / "fit_results.json"
    )

    assert runcard_src.is_file(), "Expected test runcard not found"
    assert precomputed_file.is_file(), "Expected precomputed results not found"

    # Prepare a copy of the runcard with absolute paths and a temp results folder.
    rc = _load_yaml(runcard_src)

    rc["data_path"] = str((base_dir / "test_commondata").resolve())
    rc["theory_path"] = str((base_dir / "test_theory").resolve())
    rc["result_path"] = str(tmp_path / "fit_results")
    # Keep the same result_ID so we can also compare with the precomputed folder name
    result_id = rc.get("result_ID", "ultranest_fit_lin_glob")

    # Write the adjusted runcard
    runcard_dst = tmp_path / "ultranest_fit_lin_glob.yaml"
    _dump_yaml(runcard_dst, rc)

    # Run the CLI: smefit NS <runcard>
    runner = CliRunner()
    result = runner.invoke(
        base_command, ["NS", str(runcard_dst)], catch_exceptions=False
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    # Load our produced results
    produced = tmp_path / "fit_results" / result_id / "fit_results.json"
    assert produced.is_file(), "fit_results.json was not produced"

    with open(produced, encoding="utf-8") as f:
        got = json.load(f)
    with open(precomputed_file, encoding="utf-8") as f:
        exp = json.load(f)

    # Compare content
    # - free parameter names (set equality, order-insensitive)
    assert set(got["free_parameters"]) == set(
        exp["free_parameters"]
    ), "Free parameters mismatch"

    # Use absolute tolerances for global scalars; evidence and max log-likelihood can shift slightly.
    np.testing.assert_allclose(
        got["max_loglikelihood"], exp["max_loglikelihood"], atol=0.8
    )
    np.testing.assert_allclose(got["logz"], exp["logz"], atol=0.8)

    # Compare posterior quantiles instead of mean/std to be robust to tails and RNG differences
    for name in exp["samples"].keys():
        assert name in got["samples"], f"Missing samples for {name}"
        q_got = _quantiles(got["samples"][name])
        q_exp = _quantiles(exp["samples"][name])
        np.testing.assert_allclose(q_got, q_exp, rtol=0.2, atol=0.1)


def test_cli_ultranest_quad_fit_matches_precomputed(tmp_path: pathlib.Path):
    # Paths to fixtures and runcard
    # The fixtures (runcard, data, theory, expected results) live under tests/fit_tests
    base_dir = pathlib.Path(__file__).parent / "fit_tests"
    runcard_src = base_dir / "ultranest_fit_quad_glob.yaml"
    precomputed_file = (
        base_dir / "test_results" / "ultranest_fit_quad_glob" / "fit_results.json"
    )

    assert runcard_src.is_file(), "Expected test runcard not found"
    assert precomputed_file.is_file(), "Expected precomputed results not found"

    # Prepare a copy of the runcard with absolute paths and a temp results folder.
    rc = _load_yaml(runcard_src)

    rc["data_path"] = str((base_dir / "test_commondata").resolve())
    rc["theory_path"] = str((base_dir / "test_theory").resolve())
    rc["result_path"] = str(tmp_path / "fit_results")
    # Keep the same result_ID so we can also compare with the precomputed folder name
    result_id = rc.get("result_ID", "ultranest_fit_quad_glob")

    # Write the adjusted runcard
    runcard_dst = tmp_path / "ultranest_fit_quad_glob.yaml"
    _dump_yaml(runcard_dst, rc)

    # Run the CLI: smefit NS <runcard>
    runner = CliRunner()
    result = runner.invoke(
        base_command, ["NS", str(runcard_dst)], catch_exceptions=False
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    # Load our produced results
    produced = tmp_path / "fit_results" / result_id / "fit_results.json"
    assert produced.is_file(), "fit_results.json was not produced"

    with open(produced, encoding="utf-8") as f:
        got = json.load(f)
    with open(precomputed_file, encoding="utf-8") as f:
        exp = json.load(f)

    # Compare content
    # - free parameter names (set equality, order-insensitive)
    assert set(got["free_parameters"]) == set(
        exp["free_parameters"]
    ), "Free parameters mismatch"

    # Use absolute tolerances for global scalars; evidence and max log-likelihood can shift slightly.
    np.testing.assert_allclose(
        got["max_loglikelihood"], exp["max_loglikelihood"], atol=0.8
    )
    np.testing.assert_allclose(got["logz"], exp["logz"], atol=0.8)

    # Compare posterior quantiles instead of mean/std to be robust to tails and RNG differences
    for name in exp["samples"].keys():
        assert name in got["samples"], f"Missing samples for {name}"
        q_got = _quantiles(got["samples"][name])
        q_exp = _quantiles(exp["samples"][name])
        np.testing.assert_allclose(q_got, q_exp, rtol=0.2, atol=0.1)


def test_cli_ultranest_lin_fit_with_rge_matches_precomputed(tmp_path: pathlib.Path):
    # Paths to fixtures and runcard
    base_dir = pathlib.Path(__file__).parent / "fit_tests"
    runcard_src = base_dir / "ultranest_fit_lin_glob_with_rge.yaml"
    precomputed_file = (
        base_dir
        / "test_results"
        / "ultranest_fit_lin_glob_with_rge"
        / "fit_results.json"
    )
    rge_matrix_file = (
        base_dir / "test_results" / "ultranest_fit_lin_glob_with_rge" / "rge_matrix.pkl"
    )

    assert runcard_src.is_file(), "Expected test runcard not found"
    assert precomputed_file.is_file(), "Expected precomputed results not found"
    assert rge_matrix_file.is_file(), "Expected RGE matrix not found"

    # Prepare a copy of the runcard with absolute paths and a temp results folder.
    rc = _load_yaml(runcard_src)
    rc["data_path"] = str((base_dir / "test_commondata").resolve())
    rc["theory_path"] = str((base_dir / "test_theory").resolve())
    rc["result_path"] = str(tmp_path / "fit_results")
    result_id = rc.get("result_ID", "ultranest_fit_lin_glob_with_rge")

    runcard_dst = tmp_path / "ultranest_fit_lin_glob_with_rge.yaml"
    _dump_yaml(runcard_dst, rc)

    # Run the CLI: smefit NS <runcard>
    runner = CliRunner()
    result = runner.invoke(
        base_command, ["NS", str(runcard_dst)], catch_exceptions=False
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    # Load produced outputs
    produced_fit = tmp_path / "fit_results" / result_id / "fit_results.json"
    produced_rge = tmp_path / "fit_results" / result_id / "rge_matrix.pkl"
    assert produced_fit.is_file(), "fit_results.json was not produced"
    assert produced_rge.is_file(), "rge_matrix.pkl was not produced"

    with open(produced_fit, encoding="utf-8") as f:
        got = json.load(f)
    with open(precomputed_file, encoding="utf-8") as f:
        exp = json.load(f)

    with open(rge_matrix_file, "rb") as f:
        rge_matrix_precomp = pickle.load(f)
    with open(produced_rge, "rb") as f:
        rge_matrix_produced = pickle.load(f)

    # Compare RGE matrices (lists of pandas.DataFrame)
    assert isinstance(rge_matrix_produced, list)
    assert isinstance(rge_matrix_precomp, list)
    assert len(rge_matrix_produced) == len(rge_matrix_precomp)
    for got_df, exp_df in zip(rge_matrix_produced, rge_matrix_precomp):
        assert list(got_df.index) == list(exp_df.index)
        assert list(got_df.columns) == list(exp_df.columns)
        np.testing.assert_allclose(got_df.values, exp_df.values, rtol=1e-4)

    # Compare deterministic-ish content with loose tolerances
    assert set(got["free_parameters"]) == set(
        exp["free_parameters"]
    )  # order-insensitive

    np.testing.assert_allclose(
        got["max_loglikelihood"], exp["max_loglikelihood"], atol=0.8
    )
    np.testing.assert_allclose(got["logz"], exp["logz"], atol=0.8)

    for name in exp["samples"].keys():
        assert name in got["samples"], f"Missing samples for {name}"
        q_got = _quantiles(got["samples"][name])
        q_exp = _quantiles(exp["samples"][name])
        np.testing.assert_allclose(q_got, q_exp, rtol=0.2, atol=0.1)


def test_cli_ultranest_quad_fit_with_rge_matches_precomputed(tmp_path: pathlib.Path):
    # Paths to fixtures and runcard
    base_dir = pathlib.Path(__file__).parent / "fit_tests"
    runcard_src = base_dir / "ultranest_fit_quad_glob_with_rge.yaml"
    precomputed_file = (
        base_dir
        / "test_results"
        / "ultranest_fit_quad_glob_with_rge"
        / "fit_results.json"
    )
    rge_matrix_file = (
        base_dir
        / "test_results"
        / "ultranest_fit_quad_glob_with_rge"
        / "rge_matrix.pkl"
    )

    assert runcard_src.is_file(), "Expected test runcard not found"
    assert precomputed_file.is_file(), "Expected precomputed results not found"
    assert rge_matrix_file.is_file(), "Expected RGE matrix not found"

    # Prepare a copy of the runcard with absolute paths and a temp results folder.
    rc = _load_yaml(runcard_src)
    rc["data_path"] = str((base_dir / "test_commondata").resolve())
    rc["theory_path"] = str((base_dir / "test_theory").resolve())
    rc["result_path"] = str(tmp_path / "fit_results")
    result_id = rc.get("result_ID", "ultranest_fit_quad_glob_with_rge")

    runcard_dst = tmp_path / "ultranest_fit_quad_glob_with_rge.yaml"
    _dump_yaml(runcard_dst, rc)

    # Run the CLI: smefit NS <runcard>
    runner = CliRunner()
    result = runner.invoke(
        base_command, ["NS", str(runcard_dst)], catch_exceptions=False
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    # Load produced outputs
    produced_fit = tmp_path / "fit_results" / result_id / "fit_results.json"
    produced_rge = tmp_path / "fit_results" / result_id / "rge_matrix.pkl"
    assert produced_fit.is_file(), "fit_results.json was not produced"
    assert produced_rge.is_file(), "rge_matrix.pkl was not produced"

    with open(produced_fit, encoding="utf-8") as f:
        got = json.load(f)
    with open(precomputed_file, encoding="utf-8") as f:
        exp = json.load(f)

    with open(rge_matrix_file, "rb") as f:
        rge_matrix_precomp = pickle.load(f)
    with open(produced_rge, "rb") as f:
        rge_matrix_produced = pickle.load(f)

    # Compare RGE matrices (lists of pandas.DataFrame)
    assert isinstance(rge_matrix_produced, list)
    assert isinstance(rge_matrix_precomp, list)
    assert len(rge_matrix_produced) == len(rge_matrix_precomp)
    for got_df, exp_df in zip(rge_matrix_produced, rge_matrix_precomp):
        assert list(got_df.index) == list(exp_df.index)
        assert list(got_df.columns) == list(exp_df.columns)
        np.testing.assert_allclose(got_df.values, exp_df.values, rtol=1e-4)

    # Compare deterministic-ish content with loose tolerances
    assert set(got["free_parameters"]) == set(
        exp["free_parameters"]
    )  # order-insensitive

    np.testing.assert_allclose(
        got["max_loglikelihood"], exp["max_loglikelihood"], atol=0.8
    )
    np.testing.assert_allclose(got["logz"], exp["logz"], atol=0.8)

    for name in exp["samples"].keys():
        assert name in got["samples"], f"Missing samples for {name}"
        q_got = _quantiles(got["samples"][name])
        q_exp = _quantiles(exp["samples"][name])
        np.testing.assert_allclose(q_got, q_exp, rtol=0.2, atol=0.1)
