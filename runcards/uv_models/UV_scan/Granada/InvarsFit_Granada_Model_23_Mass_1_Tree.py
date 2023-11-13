import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=23, collection="Granada", mass=1, pto="NLO", eft="NHO" )


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
	gWqf33 = results.gWqf33
	return np.abs(gWqf33)

def inv5(results):
	gWqf33 = results.gWqf33
	gWH = results.gWH
	return (gWqf33*gWH)/np.abs(gWqf33)

