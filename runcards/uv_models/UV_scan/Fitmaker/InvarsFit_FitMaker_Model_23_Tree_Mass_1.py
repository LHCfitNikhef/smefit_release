import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=23,
	 collection= "Fitmaker",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	gWH = results.gWH
	return np.abs(gWH)

def build_uv_posterior(results):
	results["gWH"] = -2*np.emath.sqrt(0.6666666666666666)*np.emath.sqrt(results.Opd)
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
