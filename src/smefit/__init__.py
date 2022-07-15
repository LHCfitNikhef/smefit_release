# -*- coding: utf-8 -*-
from .analyze import run_report
from .postfit import Postfit
from .runner import Runner


def run(runcard_folder, mode, fit_card, replica=None):
    """
    Run the |SMEFiT| package

    Parameters
    ----------
        runcard_folder : pathlib.Path
            path to runcard folder
            Note also the results will be stored in this folder.
        mode: "NS"
            running mode: "NS" = Nested Sampling
        fit_card: dict
           fit run card
    """
    # run NS
    if mode == "NS":
        runner = Runner.from_file(runcard_folder, fit_card)
        runner.ns()
    elif mode == "MC":
        if replica is not None:
            runner = Runner.from_file(runcard_folder, fit_card, replica)
            runner.mc()
        else:
            raise ValueError(
                "Montecarlo method requires to select replica number. Usage: -n replica_number"
            )
    elif mode == "PF":
        if replica is not None:
            postfit = Postfit.from_file(runcard_folder, fit_card)
            postfit.save(replica)
        else:
            raise ValueError(
                "PostFit method requires to select the number of replicas. Usage: -n replicas_number"
            )
    elif mode == "R":
        run_report(runcard_folder, fit_card)
    else:
        raise NotImplementedError(
            f"MODE={mode} is not valid, chose between NS, MC, PF and R."
        )
