import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=21,
	 collection= "Fitmaker",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	gBH = results.gBH
	return np.abs(gBH)

def build_uv_posterior(results):
	results["gBH"] = -(np.emath.sqrt(2)*np.emath.sqrt(results.Opd))
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
