import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id= "Varphi_massbound",
	 collection= "1LoopMatching",
	 mass=1, # This is a test version in which we don't fix the mass but fit it.
	 pto="NLO",
	 eft="NHO"
)

def inv1(results):
	lamvarphi = results.lamvarphi
	return np.abs(lamvarphi)

def inv2(results):
	lamvarphi = results.lamvarphi
	yVarphiuf33 = results.yVarphiuf33
	return (lamvarphi*yVarphiuf33)/np.abs(lamvarphi)

def inv3(results):
	Mvarphi = results.m
	return np.abs(Mvarphi)

