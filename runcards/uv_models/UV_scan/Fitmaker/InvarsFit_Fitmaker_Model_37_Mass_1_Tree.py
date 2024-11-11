import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=37, collection="Fitmaker", mass=1, pto="NLO", eft="NHO" )


def inv1(results):
	lamNef1 = results.lamNef1
	return np.abs(lamNef1)

