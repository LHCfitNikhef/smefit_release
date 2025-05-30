# -*- coding: utf-8 -*-
import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=21, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")


def inv1(results):
    gBH = results.gBH
    return np.abs(gBH)
