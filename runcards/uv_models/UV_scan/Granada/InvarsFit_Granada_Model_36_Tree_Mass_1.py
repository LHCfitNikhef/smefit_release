import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=36,
	 collection= "Granada",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	gY5f33 = results.gY5f33
	return np.abs(gY5f33)

def build_uv_posterior(results):
	results["gY5f33"] = -(np.emath.sqrt(1.5)*np.emath.sqrt(results.OQt1))
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
