import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=2,
	 collection= "Granada",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	kS = results.kS
	return np.abs(kS)

def build_uv_posterior(results):
	results["kS"] = -(np.emath.sqrt(2)*np.emath.sqrt(results.Opd))
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
