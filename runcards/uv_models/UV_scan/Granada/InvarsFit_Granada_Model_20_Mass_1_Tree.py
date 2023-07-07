# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=20, collection="Granada", mass=1, pto="NLO", eft="NHO")  # in TeV


def inv1(results):
    yPhiquf33 = results.yPhiquf33
    return np.abs(yPhiquf33)


def build_uv_posterior(results):
    results["yPhiquf33"] = ((0 - 3j) * np.emath.sqrt(results.OQt1)) / np.emath.sqrt(2)
    return results


def check_constrain(wc, uv):
    pass


inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
