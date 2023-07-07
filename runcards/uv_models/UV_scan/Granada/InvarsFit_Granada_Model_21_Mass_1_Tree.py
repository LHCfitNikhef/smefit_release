# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=21, collection="Granada", mass=1, pto="NLO", eft="NHO")  # in TeV


def inv1(results):
    gBdf11 = results.gBdf11
    return np.abs(gBdf11)


def inv2(results):
    gBef11 = results.gBef11
    return np.abs(gBef11)


def inv3(results):
    gBef22 = results.gBef22
    return np.abs(gBef22)


def inv4(results):
    gBef33 = results.gBef33
    return np.abs(gBef33)


def inv5(results):
    gBdf11 = results.gBdf11
    gBH = results.gBH
    return (gBdf11 * gBH) / np.abs(gBdf11)


def inv6(results):
    gBH = results.gBH
    gBqf11 = results.gBqf11
    return (gBH * gBqf11) / np.abs(gBH)


def inv7(results):
    gBH = results.gBH
    gBqf33 = results.gBqf33
    return (gBH * gBqf33) / np.abs(gBH)


def inv8(results):
    gBH = results.gBH
    gBuf11 = results.gBuf11
    return (gBH * gBuf11) / np.abs(gBH)


def inv9(results):
    gBH = results.gBH
    gBuf33 = results.gBuf33
    return (gBH * gBuf33) / np.abs(gBH)


def build_uv_posterior(results):
    results["gBdf11"] = (
        (0.0 - 1.4142135623730951j) * results.O1qd * np.emath.sqrt(results.Ott1)
    ) / results.OQt1
    results["gBef11"] = (
        (0.0 - 1.4142135623730951j) * results.Ope * np.emath.sqrt(results.Ott1)
    ) / results.Opt
    results["gBef22"] = (
        (0.0 - 1.4142135623730951j) * results.Opmu * np.emath.sqrt(results.Ott1)
    ) / results.Opt
    results["gBef33"] = (
        (0.0 - 1.4142135623730951j) * results.Opta * np.emath.sqrt(results.Ott1)
    ) / results.Opt
    results["gBH"] = ((0.0 - 0.7071067811865476j) * results.Opt) / np.emath.sqrt(
        results.Ott1
    )
    results["gBqf11"] = (
        (0.0 - 1.4142135623730951j) * results.OpqMi * np.emath.sqrt(results.Ott1)
    ) / results.Opt
    results["gBqf33"] = ((0.0 - 0.7071067811865476j) * results.OQt1) / np.emath.sqrt(
        results.Ott1
    )
    results["gBuf11"] = (
        (0.0 - 1.4142135623730951j) * results.Opui * np.emath.sqrt(results.Ott1)
    ) / results.Opt
    results["gBuf33"] = (0.0 - 1.4142135623730951j) * np.emath.sqrt(results.Ott1)
    return results


def check_constrain(wc, uv):
    pass


inspect_model(
    MODEL_SPECS,
    build_uv_posterior,
    [inv1, inv2, inv3, inv4, inv5, inv6, inv7, inv8, inv9],
    check_constrain,
)
