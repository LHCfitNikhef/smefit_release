import numpy as np
from utils import inspect_model

MODEL_SPECS = dict(id="Q1_Q7_W_Simple", collection="Multiparticle", mass=332, pto="NLO", eft="NHO" )


def inv1(results):
	gWtiH = results.gWtiH
	return np.abs(gWtiH)

def inv2(results):
	lamQuf = results.lamQuf
	return np.abs(lamQuf)

