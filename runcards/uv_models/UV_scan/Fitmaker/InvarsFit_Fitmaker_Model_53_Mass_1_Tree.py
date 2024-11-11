import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=53, collection="Fitmaker", mass=1, pto="NLO", eft="NHO" )


def inv1(results):
	gHB = results.gHB
	return np.abs(gHB)

