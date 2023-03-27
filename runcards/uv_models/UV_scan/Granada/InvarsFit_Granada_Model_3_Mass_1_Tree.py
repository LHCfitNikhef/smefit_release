import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=3,
	 collection= "Granada",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	yS1f12 = results.yS1f12
	yS1f21 = results.yS1f21
	return yS1f12*yS1f21

def build_uv_posterior(results):
	results["AA"] = AA
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
