# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=27, collection="Granada", mass=1, pto="NLO", eft="NHO")  # in TeV


def inv1(results):
    gHf33 = results.gHf33
    return np.abs(gHf33)


def build_uv_posterior(results):
    results["gHf33"] = ((0 - 3j) * np.emath.sqrt(results.OQQ1)) / np.emath.sqrt(2)
    return results


def check_constrain(wc, uv):
    pass


inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
