import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=50,
	 collection= "Fitmaker",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	sLt = results.sLt
	v = results.v
	return sLt**2/v**2

def build_uv_posterior(results):
	results["sLt"] = -(np.emath.sqrt(results.OpQM)*v)
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
