import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=45,
	 collection= "Granada",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	lamQ1uf3 = results.lamQ1uf3
	return np.abs(lamQ1uf3)

def build_uv_posterior(results):
	results["lamQ1uf3"] = (0-1J)*np.emath.sqrt(2)*np.emath.sqrt(results.Opt)
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
