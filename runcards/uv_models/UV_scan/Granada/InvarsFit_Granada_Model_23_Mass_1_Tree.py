# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=23, collection="Granada", mass=1, pto="NLO", eft="NHO")  # in TeV


def inv1(results):
    gWH = results.gWH
    return np.abs(gWH)


def inv2(results):
    gWH = results.gWH
    gWLf11 = results.gWLf11
    return (gWH * gWLf11) / np.abs(gWH)


def inv3(results):
    gWH = results.gWH
    gWLf22 = results.gWLf22
    return (gWH * gWLf22) / np.abs(gWH)


def inv4(results):
    gWH = results.gWH
    gWLf33 = results.gWLf33
    return (gWH * gWLf33) / np.abs(gWH)


def inv5(results):
    gWH = results.gWH
    gWqf11 = results.gWqf11
    return (gWH * gWqf11) / np.abs(gWH)


def inv6(results):
    gWH = results.gWH
    gWqf33 = results.gWqf33
    return (gWH * gWqf33) / np.abs(gWH)


def build_uv_posterior(results):
    results["gWH"] = (
        (0.0 - 2.0j) * np.emath.sqrt(results.O3pl1) * np.emath.sqrt(results.O3pl2)
    ) / np.emath.sqrt(results.Oll)
    results["gWLf11"] = (
        (0.0 - 2.0j) * np.emath.sqrt(results.O3pl1) * np.emath.sqrt(results.Oll)
    ) / np.emath.sqrt(results.O3pl2)
    results["gWLf22"] = (
        (0.0 - 2.0j) * np.emath.sqrt(results.O3pl2) * np.emath.sqrt(results.Oll)
    ) / np.emath.sqrt(results.O3pl1)
    results["gWLf33"] = ((0.0 - 2.0j) * results.O3pl3 * np.emath.sqrt(results.Oll)) / (
        np.emath.sqrt(results.O3pl1) * np.emath.sqrt(results.O3pl2)
    )
    results["gWqf11"] = ((0.0 + 2.0j) * np.emath.sqrt(results.Oll) * results.OpqMi) / (
        np.emath.sqrt(results.O3pl1) * np.emath.sqrt(results.O3pl2)
    )
    results["gWqf33"] = -3.4641016151377544 * np.emath.sqrt(results.OQQ1)
    return results


def check_constrain(wc, uv):
    pass


inspect_model(
    MODEL_SPECS,
    build_uv_posterior,
    [inv1, inv2, inv3, inv4, inv5, inv6],
    check_constrain,
)
