# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=38, collection="Granada", mass=1, pto="NLO", eft="NHO")  # in TeV


def inv1(results):
    lamEff3 = results.lamEff3
    return np.abs(lamEff3)


def build_uv_posterior(results):
    results["lamEff3"] = (0 - 2j) * np.emath.sqrt(results.Opl3)
    return results


def check_constrain(wc, uv):
    pass


inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
