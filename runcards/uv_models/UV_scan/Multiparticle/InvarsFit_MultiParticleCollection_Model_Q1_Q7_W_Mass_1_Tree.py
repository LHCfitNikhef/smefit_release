import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id="Q1_Q7_W", collection="MultiParticleCollection", mass=1, pto="NLO", eft="NHO" )


def inv1(results):
	gWLf11 = results.gWLf11
	return np.abs(gWLf11)

def inv2(results):
	gWLf33 = results.gWLf33
	return np.abs(gWLf33)

def inv3(results):
	gWqf33 = results.gWqf33
	return np.abs(gWqf33)

def inv4(results):
	gWqf33 = results.gWqf33
	gWtiH = results.gWtiH
	return (gWqf33*gWtiH)/np.abs(gWqf33)

def inv5(results):
	lamQ1uf3 = results.lamQ1uf3
	return np.abs(lamQ1uf3)

def inv6(results):
	lamQ7f33 = results.lamQ7f33
	return np.abs(lamQ7f33)

