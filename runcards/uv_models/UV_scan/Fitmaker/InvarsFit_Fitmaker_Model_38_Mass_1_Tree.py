# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=38, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")  # in TeV


def inv1(results):
    lamEff1 = results.lamEff1
    return np.abs(lamEff1)


def build_uv_posterior(results):
    results["lamEff1"] = (0 - 2j) * np.emath.sqrt(results.Opl1)
    return results


def check_constrain(wc, uv):
    pass


inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
