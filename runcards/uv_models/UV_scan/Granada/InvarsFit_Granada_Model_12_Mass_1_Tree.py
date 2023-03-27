import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=12,
	 collection= "Granada",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	yomega4uuf33 = results.yomega4uuf33
	return np.abs(yomega4uuf33)

def build_uv_posterior(results):
	results["y[\[Omega]4][uu][1, 3, 3]"] = -np.emath.sqrt(results.Ott1)
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
