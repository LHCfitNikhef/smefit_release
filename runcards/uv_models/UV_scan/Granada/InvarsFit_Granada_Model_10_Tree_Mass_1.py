import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=10,
	 collection= "Granada",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	yomega1qqf33 = results.yomega1qqf33
	return np.abs(yomega1qqf33)

def build_uv_posterior(results):
	results["yomega1qqf33"] = -0.5*(np.emath.sqrt(3)*np.emath.sqrt(results.OQQ1))
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
