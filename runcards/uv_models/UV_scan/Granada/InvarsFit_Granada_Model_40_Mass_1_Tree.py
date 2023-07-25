# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=40, collection="Granada", mass=1, pto="NLO", eft="NHO")  # in TeV


def inv1(results):
    lamDelta3f3 = results.lamDelta3f3
    return np.abs(lamDelta3f3)


def build_uv_posterior(results):
    results["lamDelta3f3"] = (0 - 1j) * np.emath.sqrt(2) * np.emath.sqrt(results.Opta)
    return results


def check_constrain(wc, uv):
    pass


inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
