import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=52,
	 collection= "Fitmaker",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	lambdaQ = results.lambdaQ
	return np.abs(lambdaQ)

def build_uv_posterior(results):
	results["lambdaQ"] = -((np.emath.sqrt(2.8504283398934938)*np.emath.sqrt(results.Otp))/2**0.75)
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
