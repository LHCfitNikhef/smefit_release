# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=44, collection="Granada", mass=1, pto="NLO", eft="NHO")  # in TeV


def inv1(results):
    lamDff3 = results.lamDff3
    return np.abs(lamDff3)


def build_uv_posterior(results):
    results["lamDff3"] = (0 - 2j) * np.emath.sqrt(results.O3pQ3)
    return results


def check_constrain(wc, uv):
    pass


inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
