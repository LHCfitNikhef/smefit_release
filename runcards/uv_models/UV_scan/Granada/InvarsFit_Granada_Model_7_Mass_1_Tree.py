# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=7, collection="Granada", mass=1, pto="NLO", eft="NHO")  # in TeV


def inv1(results):
    kXi1 = results.kXi1
    return np.abs(kXi1)


def build_uv_posterior(results):
    results["kXi1"] = ((0 - 1j) * np.emath.sqrt(results.Opd)) / np.emath.sqrt(2)
    return results


def check_constrain(wc, uv):
    pass


inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
