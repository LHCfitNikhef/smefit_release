import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=51,
	 collection= "Fitmaker",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	sRt = results.sRt
	return np.abs(sRt)

def build_uv_posterior(results):
	results["sRt"] = (0-0.24622J)*np.emath.sqrt(results.Opt)
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
