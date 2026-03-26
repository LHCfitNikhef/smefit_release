# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import pytest

from smefit.external_chi2 import load_external_chi2


def test_load_external_chi2_with_fake_external_module(monkeypatch):
    """Use the provided fake external chi2 module under tests/fake_external_chi2
    to ensure load_external_chi2 imports it and returns a callable that computes
    the L2 norm of the provided coefficient array."""

    module_path = Path(__file__).parent / "fake_external_chi2" / "test_ext_chi2.py"

    # Make sure Python can import the module by adding its parent to sys.path
    monkeypatch.syspath_prepend(str(module_path.parent))

    external_chi2 = {"ExternalChi2": {"path": str(module_path)}}

    coefficients = {"dummy": 1}
    rge_dict = None

    loaded = load_external_chi2(external_chi2, coefficients, rge_dict)

    assert isinstance(loaded, list)
    assert len(loaded) == 1

    compute_fn = loaded[0]

    vals = np.array([1.0, 2.0, 3.0])
    result = compute_fn(vals)

    assert result == pytest.approx(np.sum(vals**2))
