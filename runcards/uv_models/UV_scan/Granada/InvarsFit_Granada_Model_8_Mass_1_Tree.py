import numpy as np

from utils import inspect_model


MODEL_SPECS = dict(
	 id=8,
	 collection= "Granada",
	 mass=1 # in TeV
	 pto="NLO",
	 eft="NHO"
)


def check_constrain(wc, uv):
	pass

inspect_model(MODEL_SPECS, build_uv_posterior, [ ], check_constrain)
