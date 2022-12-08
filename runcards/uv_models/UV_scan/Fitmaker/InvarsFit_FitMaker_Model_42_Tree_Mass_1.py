import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=42,
	 collection= "Fitmaker",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	lamSigma1f1 = results.lamSigma1f1
	return np.abs(lamSigma1f1)

def build_uv_posterior(results):
	results["lamSigma1f1"] = ((0-4J)*np.emath.sqrt(results.Opl1))/np.emath.sqrt(3)
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
