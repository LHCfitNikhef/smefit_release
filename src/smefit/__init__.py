# -*- coding: utf-8 -*-
from .runner import Runner


def run(root_path, mode, fit_card):
    """
    Run the |SMEFiT| package

    Parameters
    ----------
        root_path : pathlib.Path
            root path where data tables (Commondata and theory) are located.
            Note also the results will be stored in this folder.
        mode: "NS"
            running mode: "NS" = Nested Sampling
        fit_card: dict
           fit run card
    """
    runner = Runner(root_path)
    # run NS
    if mode == "NS":
        runner.ns(fit_card)
    else:
        raise NotImplementedError(
            f"MODE={mode} is not valid, the only implemented feature atm is NS"
        )
