import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=47,
	 collection= "Fitmaker",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	lamQ7f1 = results.lamQ7f1
	return np.abs(lamQ7f1)

def build_uv_posterior(results):
	results["lamQ7f1"] = -(np.emath.sqrt(2)*np.emath.sqrt(results.Opui))
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
