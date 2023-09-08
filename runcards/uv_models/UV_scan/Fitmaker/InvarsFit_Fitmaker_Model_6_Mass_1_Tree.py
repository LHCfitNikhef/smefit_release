import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=6, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")


def inv1(results):
    kXi = results.kXi
    return np.abs(kXi)
