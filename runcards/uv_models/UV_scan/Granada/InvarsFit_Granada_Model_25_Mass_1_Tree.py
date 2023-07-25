# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=25, collection="Granada", mass=1, pto="NLO", eft="NHO")  # in TeV


def inv1(results):
    gGdf11 = results.gGdf11
    return np.abs(gGdf11)


def inv2(results):
    gGdf11 = results.gGdf11
    gGqf33 = results.gGqf33
    return (gGdf11 * gGqf33) / np.abs(gGdf11)


def inv3(results):
    gGdf11 = results.gGdf11
    gGuf33 = results.gGuf33
    return (gGdf11 * gGuf33) / np.abs(gGdf11)


def build_uv_posterior(results):
    results["gGdf11"] = (
        (0 - 1j) * np.emath.sqrt(6) * results.O8qd * np.emath.sqrt(results.Ott1)
    ) / results.OQt8
    results["gGqf33"] = ((0 - 1j) * results.OQt8) / (
        np.emath.sqrt(6) * np.emath.sqrt(results.Ott1)
    )
    results["gGuf33"] = (0 - 1j) * np.emath.sqrt(6) * np.emath.sqrt(results.Ott1)
    return results


def check_constrain(wc, uv):
    pass


inspect_model(MODEL_SPECS, build_uv_posterior, [inv1, inv2, inv3], check_constrain)
