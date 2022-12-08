import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=50,
	 collection= "Fitmaker",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	sLt = results.sLt
	return np.abs(sLt)

def build_uv_posterior(results):
	results["sLt"] = (-12311*np.emath.sqrt(results.OpQM))/50000.
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
