import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=33,
	 collection= "Granada",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	gQv5uqf33 = results.gQv5uqf33
	return np.abs(gQv5uqf33)

def build_uv_posterior(results):
	results["gQv5uqf33"] = -(np.emath.sqrt(1.5)*np.emath.sqrt(results.OQt1))
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
