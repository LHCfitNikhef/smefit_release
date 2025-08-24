# -*- coding: utf-8 -*-
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


def _quantiles(a, qs=(0.025, 0.975)) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    return np.quantile(arr, qs)


def _ci_atol(width: float) -> float:
    # Adaptive atol for 95% CI endpoints
    return max(0.02, min(0.1, 0.05 * float(width)))


@pytest.mark.parametrize(
    "runcard_filename,result_id,expect_rge",
    [
        ("ultranest_fit_lin_glob.yaml", "ultranest_fit_lin_glob", False),
        ("ultranest_fit_quad_glob.yaml", "ultranest_fit_quad_glob", False),
        (
            "ultranest_fit_lin_glob_with_rge.yaml",
            "ultranest_fit_lin_glob_with_rge",
            True,
        ),
        (
            "ultranest_fit_quad_glob_with_rge.yaml",
            "ultranest_fit_quad_glob_with_rge",
            True,
        ),
    ],
    ids=["lin", "quad", "lin_rge", "quad_rge"],
)
def test_cli_ultranest_fit_matches_precomputed(
    tmp_path: pathlib.Path, runcard_filename: str, result_id: str, expect_rge: bool
):
    base_dir = pathlib.Path(__file__).parent / "fit_tests"
    runcard_src = base_dir / runcard_filename
    precomputed_dir = base_dir / "test_results" / result_id
    precomputed_file = precomputed_dir / "fit_results.json"
    rge_matrix_file = precomputed_dir / "rge_matrix.pkl"

    assert runcard_src.is_file()
    assert precomputed_file.is_file()
    if expect_rge:
        assert rge_matrix_file.is_file()

    # Prepare runcard with absolute paths and tmp result_path
    rc = _load_yaml(runcard_src)
    rc["data_path"] = str((base_dir / "test_commondata").resolve())
    rc["theory_path"] = str((base_dir / "test_theory").resolve())
    rc["result_path"] = str(tmp_path / "fit_results")
    rc.setdefault("result_ID", result_id)

    runcard_dst = tmp_path / runcard_src.name
    _dump_yaml(runcard_dst, rc)

    # Run CLI
    runner = CliRunner()
    result = runner.invoke(
        base_command, ["NS", str(runcard_dst)], catch_exceptions=False
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    # Load produced outputs
    produced_fit = tmp_path / "fit_results" / result_id / "fit_results.json"
    assert produced_fit.is_file(), "fit_results.json was not produced"

    with open(produced_fit, encoding="utf-8") as f:
        got = json.load(f)
    with open(precomputed_file, encoding="utf-8") as f:
        exp = json.load(f)

    # Optional: RGE matrix compare
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

    # Compare deterministic-ish content
    assert set(got["free_parameters"]) == set(exp["free_parameters"])
    np.testing.assert_allclose(
        got["max_loglikelihood"], exp["max_loglikelihood"], atol=0.5
    )
    np.testing.assert_allclose(got["logz"], exp["logz"], atol=0.5)

    # Per-parameter 95% CI endpoints with adaptive tol
    for name in exp["samples"].keys():
        assert name in got["samples"], f"Missing samples for {name}"
        q_got = _quantiles(got["samples"][name])
        q_exp = _quantiles(exp["samples"][name])
        atol = _ci_atol(q_exp[1] - q_exp[0])
        np.testing.assert_allclose(q_got, q_exp, rtol=0.1, atol=atol)


@pytest.mark.parametrize(
    "runcard_filename,result_id",
    [
        ("ultranest_fit_indiv_lin.yaml", "ultranest_fit_indiv_lin"),
        ("ultranest_fit_indiv_quad.yaml", "ultranest_fit_indiv_quad"),
    ],
    ids=["indiv_lin", "indiv_quad"],
)
def test_cli_ultranest_fit_indiv_matches_precomputed(
    tmp_path: pathlib.Path, runcard_filename: str, result_id: str
):
    # Paths to fixtures and runcard
    base_dir = pathlib.Path(__file__).parent / "fit_tests"
    runcard_src = base_dir / runcard_filename
    precomputed_file = base_dir / "test_results" / result_id / "fit_results.json"

    assert runcard_src.is_file(), "Expected test runcard not found"
    assert precomputed_file.is_file(), "Expected precomputed results not found"

    # Prepare a copy of the runcard with absolute paths and a temp results folder.
    rc = _load_yaml(runcard_src)
    rc["data_path"] = str((base_dir / "test_commondata").resolve())
    rc["theory_path"] = str((base_dir / "test_theory").resolve())
    rc["result_path"] = str(tmp_path / "fit_results")
    rc.setdefault("result_ID", result_id)

    runcard_dst = tmp_path / runcard_src.name
    _dump_yaml(runcard_dst, rc)

    # Run the CLI: smefit NS <runcard>
    runner = CliRunner()
    result = runner.invoke(
        base_command, ["NS", str(runcard_dst)], catch_exceptions=False
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

    for name in exp["max_loglikelihood"].keys():
        assert name in got["max_loglikelihood"], f"Missing max_loglikelihood for {name}"
        np.testing.assert_allclose(
            got["max_loglikelihood"][name], exp["max_loglikelihood"][name], atol=0.5
        )

    for name in exp["logz"].keys():
        assert name in got["logz"], f"Missing logz for {name}"
        np.testing.assert_allclose(got["logz"][name], exp["logz"][name], atol=0.5)

    # Per-parameter 95% CI endpoints with adaptive tol
    for name in exp["samples"].keys():
        assert name in got["samples"], f"Missing samples for {name}"
        q_got = _quantiles(got["samples"][name])
        q_exp = _quantiles(exp["samples"][name])
        atol = _ci_atol(q_exp[1] - q_exp[0])
        np.testing.assert_allclose(q_got, q_exp, rtol=0.1, atol=atol)
