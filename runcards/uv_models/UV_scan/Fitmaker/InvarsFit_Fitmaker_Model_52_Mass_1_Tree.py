import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=52, collection="Fitmaker", mass=1, pto="NLO", eft="NHO")


def inv1(results):
    lambdaQ = results.lambdaQ
    return np.abs(lambdaQ)
