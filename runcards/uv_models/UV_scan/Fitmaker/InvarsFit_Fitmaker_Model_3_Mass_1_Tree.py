import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=3, collection="Fitmaker", mass=1, pto="NLO", eft="NHO" )


def inv1(results):
	yS1 = results.yS1
	return np.abs(yS1)

