import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=46,
	 collection= "Fitmaker",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	lamQ5f1 = results.lamQ5f1
	return np.abs(lamQ5f1)

def build_uv_posterior(results):
	results["lamQ5f1"] = (0-1J)*np.emath.sqrt(2)*np.emath.sqrt(results.Opdi)
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
