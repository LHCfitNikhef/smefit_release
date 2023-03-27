import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=44,
	 collection= "Fitmaker",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	lamDff1 = results.lamDff1
	return np.abs(lamDff1)

def build_uv_posterior(results):
	results["lamDff1"] = (0-2J)*np.emath.sqrt(results.O3pq)
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
