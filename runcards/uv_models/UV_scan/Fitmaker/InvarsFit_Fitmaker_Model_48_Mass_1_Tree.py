import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=48,
	 collection= "Fitmaker",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	lamT1f1 = results.lamT1f1
	return np.abs(lamT1f1)

def build_uv_posterior(results):
	results["lamT1f1"] = (0-2J)*np.emath.sqrt(results.OpqMi)
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
