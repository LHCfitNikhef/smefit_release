# -*- coding: utf-8 -*-
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
    runner = Runner(runcard_folder, fit_card)
    # run NS
    if mode == "NS":
        runner.ns()
    else:
        raise NotImplementedError(
            f"MODE={mode} is not valid, the only implemented feature atm is NS"
        )
