# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=51, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")  # in TeV


def inv1(results):
    sRt = results.sRt
    v = results.v
    return sRt**2 / v**2


def build_uv_posterior(results):
    results["sRt"] = (0 - 1j) * np.emath.sqrt(results.Opt) * v
    return results


def check_constrain(wc, uv):
    pass


inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
