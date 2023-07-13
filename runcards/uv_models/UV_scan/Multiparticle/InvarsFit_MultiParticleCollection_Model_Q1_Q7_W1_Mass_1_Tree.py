import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=Q1_Q7_W1,
	 collection= "MultiParticleCollection",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def inv1(results):
	gWtiH = results.gWtiH
	return np.abs(gWtiH)

def build_uv_posterior(results):
	results["gWtiH"] = -(np.emath.sqrt(-12311*np.emath.sqrt(2)*results.Otp + 8638*lamQ1uf(3)**2 + 8638*lamQ7f3(3)**2)/np.emath.sqrt(4319))
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1], check_constrain)
