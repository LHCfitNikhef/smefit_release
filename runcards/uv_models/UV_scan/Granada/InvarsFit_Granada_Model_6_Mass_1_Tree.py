import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=6,
	 collection= "Granada",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	kXi = results.kXi
	return np.abs(kXi)

def build_uv_posterior(results):
	results["kXi"] = (0-1J)*np.emath.sqrt(2)*np.emath.sqrt(results.Opd)
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
