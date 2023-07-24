import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=15,
	 collection= "Granada",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	yZetaqqf33 = results.yZetaqqf33
	return np.abs(yZetaqqf33)

def build_uv_posterior(results):
	results["yZetaqqf33"] = -(np.emath.sqrt(0.3)*np.emath.sqrt(results.OQQ1))
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
