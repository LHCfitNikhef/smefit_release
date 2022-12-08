import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=37,
	 collection= "Granada",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	lamNef3 = results.lamNef3
	return np.abs(lamNef3)

def build_uv_posterior(results):
	results["lamNef3"] = -2*np.emath.sqrt(results.Opl3)
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
