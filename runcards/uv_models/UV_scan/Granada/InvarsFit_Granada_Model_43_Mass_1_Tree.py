# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=43, collection="Granada", mass=1, pto="NLO", eft="NHO")  # in TeV


def inv1(results):
    lamUf3 = results.lamUf3
    return np.abs(lamUf3)


def build_uv_posterior(results):
    results["lamUf3"] = -(np.emath.sqrt(2) * np.emath.sqrt(results.OpQM))
    return results


def check_constrain(wc, uv):
    pass


inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
