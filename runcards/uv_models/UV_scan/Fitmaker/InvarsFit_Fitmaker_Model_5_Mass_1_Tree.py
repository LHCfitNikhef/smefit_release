# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=5, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")  # in TeV


def inv1(results):
    cotBeta = results.cotBeta
    return np.abs(cotBeta)


def inv2(results):
    cotBeta = results.cotBeta
    Z6 = results.Z6
    return (cotBeta * Z6) / np.abs(cotBeta)


def build_uv_posterior(results):
    results["cotBeta"] = (
        (0 - 1.4252141699467469j) * np.emath.sqrt(3) * np.emath.sqrt(results.OQt1)
    )
    results["Z6"] = ((0 - 1j) * results.Otp) / (
        np.emath.sqrt(6) * np.emath.sqrt(results.OQt1)
    )
    return results


def check_constrain(wc, uv):
    pass


inspect_model(MODEL_SPECS, build_uv_posterior, [inv1, inv2], check_constrain)
