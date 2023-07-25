# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=5, collection="Granada", mass=1, pto="NLO", eft="NHO")  # in TeV


def inv1(results):
    lamVarphi = results.lamVarphi
    return np.abs(lamVarphi)


def inv2(results):
    lamVarphi = results.lamVarphi
    yVarphiuf33 = results.yVarphiuf33
    return (lamVarphi * yVarphiuf33) / np.abs(lamVarphi)


def build_uv_posterior(results):
    results["lamVarphi"] = ((0 - 1j) * results.Otp) / (
        np.emath.sqrt(6) * np.emath.sqrt(results.OQt1)
    )
    results["yVarphiuf33"] = (0 - 1j) * np.emath.sqrt(6) * np.emath.sqrt(results.OQt1)
    return results


def check_constrain(wc, uv):
    pass


inspect_model(MODEL_SPECS, build_uv_posterior, [inv1, inv2], check_constrain)
