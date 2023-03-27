import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=18,
	 collection= "Granada",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	yOmega4f33 = results.yOmega4f33
	return np.abs(yOmega4f33)

def build_uv_posterior(results):
	results["y[\[CapitalOmega]4][1, 3, 3]"] = -(np.emath.sqrt(2)*np.emath.sqrt(results.Ott1))
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
