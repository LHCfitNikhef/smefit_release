# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=39, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")  # in TeV


def inv1(results):
    lamDelta1f1 = results.lamDelta1f1
    return np.abs(lamDelta1f1)


def build_uv_posterior(results):
    results["lamDelta1f1"] = -(np.emath.sqrt(2) * np.emath.sqrt(results.Ope))
    return results


def check_constrain(wc, uv):
    pass


inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
