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
	gWLf11 = results.gWLf11
	return np.abs(gWLf11)

def inv2(results):
	gWLf11 = results.gWLf11
	gWLf22 = results.gWLf22
	return (gWLf11*gWLf22)/np.abs(gWLf11)

def inv3(results):
	gWLf33 = results.gWLf33
	return np.abs(gWLf33)

def inv4(results):
	gWqf11 = results.gWqf11
	return np.abs(gWqf11)

def inv5(results):
	gWqf11 = results.gWqf11
	gWqf33 = results.gWqf33
	return (gWqf11*gWqf33)/np.abs(gWqf11)

def inv6(results):
	gWqf11 = results.gWqf11
	gWtiH = results.gWtiH
	return (gWqf11*gWtiH)/np.abs(gWqf11)

def inv7(results):
	lamQ1uf3 = results.lamQ1uf3
	return np.abs(lamQ1uf3)

def inv8(results):
	lamQ7f33 = results.lamQ7f33
	return np.abs(lamQ7f33)

def build_uv_posterior(results):
	results["gWLf11"] = ((0-2J)*np.emath.sqrt(results.O3pl1)*np.emath.sqrt(results.Oll))/np.emath.sqrt(results.O3pl2)
	results["gWLf22"] = ((0-2J)*np.emath.sqrt(results.O3pl2)*np.emath.sqrt(results.Oll))/np.emath.sqrt(results.O3pl1)
	results["gWLf33"] = ((0-2J)*results.O3pl3*np.emath.sqrt(results.Oll))/(np.emath.sqrt(results.O3pl1)*np.emath.sqrt(results.O3pl2))
	results["gWqf11"] = (4*results.O13qq)/(np.emath.sqrt(3)*np.emath.sqrt(results.OQQ1))
	results["gWqf33"] = -2*np.emath.sqrt(3)*np.emath.sqrt(results.OQQ1)
	results["gWtiH"] = ((0-2J)*np.emath.sqrt(results.O3pl1)*np.emath.sqrt(results.O3pl2))/np.emath.sqrt(results.Oll)
	results["lamQ1uf3"] = -np.emath.sqrt(-106342418*np.emath.sqrt(2)*results.Obp - 3610684*results.Opt + 2572999*np.emath.sqrt(2)*results.Otp)/(2.*np.emath.sqrt(902671))
	results["lamQ7f33"] = -np.emath.sqrt(-106342418*np.emath.sqrt(2)*results.Obp + 3610684*results.Opt + 2572999*np.emath.sqrt(2)*results.Otp)/(2.*np.emath.sqrt(902671))
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1, inv2, inv3, inv4, inv5, inv6, inv7, inv8], check_constrain)
