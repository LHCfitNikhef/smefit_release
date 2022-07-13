# -*- coding: utf-8 -*-
import pathlib

import click

from ..analyze import run_report
from ..postfit import Postfit
from ..runner import Runner
from .base import base_command, root_path

runcard_path = click.option(
    "-p",
    "--runcard_path",
    type=click.Path(path_type=pathlib.Path),
    default=root_path / "runcards",
    required=False,
    help="path to runcard",
)


fit_card = click.option(
    "-f",
    "--fit_card",
    type=str,
    default=None,
    required=True,
    help="fit card name",
)

n_replica = click.option(
    "-n",
    "--n_replica",
    default=None,
    required=True,
    help="Number of the replica",
)


@base_command.command("NS")
@runcard_path
@fit_card
def nested_sampling(runcard_path, fit_card):
    runner = Runner.from_file(runcard_path, fit_card)
    runner.ns()


@base_command.command("MC")
@runcard_path
@fit_card
@n_replica
def monte_carlo_fit(runcard_path, fit_card, n_replica):
    runner = Runner.from_file(runcard_path, fit_card, n_replica)
    runner.mc()


@base_command.command("PF")
@runcard_path
@fit_card
@n_replica
def post_fit(runcard_path, fit_card, n_replica):
    postfit = Postfit.from_file(runcard_path, fit_card)
    postfit.save(n_replica)


@base_command.command("R")
@runcard_path
@fit_card
def report(runcard_path, fit_card):
    run_report(runcard_path, fit_card)
