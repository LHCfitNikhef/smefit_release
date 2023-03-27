import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=39,
	 collection= "Granada",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	lamDelta1f3 = results.lamDelta1f3
	return np.abs(lamDelta1f3)

def build_uv_posterior(results):
	results["lamDelta1f3"] = -(np.emath.sqrt(2)*np.emath.sqrt(results.Opta))
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
