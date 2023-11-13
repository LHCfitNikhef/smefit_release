import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id=21, collection="Granada", mass=1, pto="NLO", eft="NHO" )

def inv1(results):
	gBH = results.gBH
	return np.abs(gBH)

def inv2(results):
	gBef11 = results.gBef11
	gBH = results.gBH
	return gBef11*gBH

def inv3(results):
	gBef22 = results.gBef22
	gBH = results.gBH
	return gBef22*gBH

def inv4(results):
	gBef33 = results.gBef33
	gBH = results.gBH
	return gBef33*gBH

def inv5(results):
	gBH = results.gBH
	gBLf11 = results.gBLf11
	return (gBH*gBLf11)/np.abs(gBH)

def inv6(results):
	gBH = results.gBH
	gBLf22 = results.gBLf22
	return gBH*gBLf22

def inv7(results):
	gBH = results.gBH
	gBLf33 = results.gBLf33
	return gBH*gBLf33

def inv8(results):
	gBH = results.gBH
	gBqf33 = results.gBqf33
	return (gBH*gBqf33)/np.abs(gBH)

def inv9(results):
	gBH = results.gBH
	gBuf33 = results.gBuf33
	return (gBH*gBuf33)/np.abs(gBH)

