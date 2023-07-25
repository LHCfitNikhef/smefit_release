# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=16, collection="Granada", mass=1, pto="NLO", eft="NHO")  # in TeV


def inv1(results):
    yOmega1qqf33 = results.yOmega1qqf33
    return np.abs(yOmega1qqf33)


def build_uv_posterior(results):
    results["yOmega1qqf33"] = -(np.emath.sqrt(3) * np.emath.sqrt(results.OQQ1))
    return results


def check_constrain(wc, uv):
    pass


inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
