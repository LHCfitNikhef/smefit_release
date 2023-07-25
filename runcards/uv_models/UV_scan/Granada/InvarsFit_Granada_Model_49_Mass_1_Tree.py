# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=49, collection="Granada", mass=1, pto="NLO", eft="NHO")  # in TeV


def inv1(results):
    lamT2f3 = results.lamT2f3
    return np.abs(lamT2f3)


def build_uv_posterior(results):
    results["lamT2f3"] = -2 * np.emath.sqrt(2) * np.emath.sqrt(results.OpQM)
    return results


def check_constrain(wc, uv):
    pass


inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
