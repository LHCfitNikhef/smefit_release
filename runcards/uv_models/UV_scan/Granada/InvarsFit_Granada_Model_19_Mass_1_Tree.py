# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=19, collection="Granada", mass=1, pto="NLO", eft="NHO")  # in TeV


def inv1(results):
    yUpsf33 = results.yUpsf33
    return np.abs(yUpsf33)


def build_uv_posterior(results):
    results["yUpsf33"] = -0.5 * (np.emath.sqrt(3) * np.emath.sqrt(results.OQQ1))
    return results


def check_constrain(wc, uv):
    pass


inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
