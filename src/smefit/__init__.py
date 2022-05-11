# -*- coding: utf-8 -*-
from .analyze import run_report
from .runner import Runner


def run(runcard_folder, mode, fit_card):
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
    elif mode == "R":
        run_report(runcard_folder, fit_card)
    else:
        raise NotImplementedError(f"MODE={mode} is not valid, chose between R, NS.")
