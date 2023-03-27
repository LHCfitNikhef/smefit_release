import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=53,
	 collection= "Fitmaker",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	gHB = results.gHB
	return np.abs(gHB)

def build_uv_posterior(results):
	results["gHB"] = -np.emath.sqrt(results.Opd)
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
