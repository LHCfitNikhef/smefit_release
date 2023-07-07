# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=24, collection="Granada", mass=1, pto="NLO", eft="NHO")  # in TeV


def inv1(results):
    gW1H = results.gW1H
    return np.abs(gW1H)


def build_uv_posterior(results):
    results["gW1H"] = -2 * np.emath.sqrt(2) * np.emath.sqrt(results.Opd)
    return results


def check_constrain(wc, uv):
    pass


inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
