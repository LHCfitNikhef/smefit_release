# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=22, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")  # in TeV


def inv1(results):
    gB1H = results.gB1H
    return np.abs(gB1H)


def build_uv_posterior(results):
    results["gB1H"] = -(np.emath.sqrt(2) * np.emath.sqrt(results.Opd))
    return results


def check_constrain(wc, uv):
    pass


inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
