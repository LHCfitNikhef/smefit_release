import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=Q1_Q7_W1_NoDegen,
	 collection= "MultiParticleCollection",
	 mass=342 # in TeV
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
	results["gWLf11"] = ((0.-5.J)*np.emath.sqrt(results.O3pl1)*np.emath.sqrt(results.Oll))/np.emath.sqrt(results.O3pl2)
	results["gWLf22"] = ((0.-5.J)*np.emath.sqrt(results.O3pl2)*np.emath.sqrt(results.Oll))/np.emath.sqrt(results.O3pl1)
	results["gWLf33"] = ((0.-5.J)*results.O3pl3*np.emath.sqrt(results.Oll))/(np.emath.sqrt(results.O3pl1)*np.emath.sqrt(results.O3pl2))
	results["gWqf11"] = (5.773502691896257*results.O13qq)/np.emath.sqrt(results.OQQ1)
	results["gWqf33"] = -8.660254037844386*np.emath.sqrt(results.OQQ1)
	results["gWtiH"] = ((0.-5.J)*np.emath.sqrt(results.O3pl1)*np.emath.sqrt(results.O3pl2))/np.emath.sqrt(results.Oll)
	results["lamQ1uf3"] = (-1.4592680567838673e-22*np.emath.sqrt(1.1182577966239109e29*results.O3pl1*results.O3pl2 - 1.7603718086737558e46*results.Obp*results.Oll - 4.226417127032306e44*results.Oll*results.Opt + 4.2592927531004275e44*results.Oll*results.Otp))/np.emath.sqrt(results.Oll)
	results["lamQ7f33"] = (-8.25384563755097e-23*np.emath.sqrt(5.7115986551446294e29*results.O3pl1*results.O3pl2 - 1.238067690061099e47*results.Obp*results.Oll + 2.9724348367301626e45*results.Oll*results.Opt + 2.995556230872537e45*results.Oll*results.Otp))/np.emath.sqrt(results.Oll)
	return results

def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [inv1, inv2, inv3, inv4, inv5, inv6, inv7, inv8], check_constrain)
