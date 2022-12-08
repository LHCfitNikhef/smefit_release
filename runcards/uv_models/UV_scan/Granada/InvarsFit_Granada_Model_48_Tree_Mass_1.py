import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=48,
	 collection= "Granada",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	lamT1f3 = results.lamT1f3
	return np.abs(lamT1f3)

def build_uv_posterior(results):
	results["lamT1f3"] = (0-2J)*np.emath.sqrt(results.OpQM)
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
