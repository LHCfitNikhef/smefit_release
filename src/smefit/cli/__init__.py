# -*- coding: utf-8 -*-
import pathlib

import click
from mpi4py import MPI

from .. import log
from ..analyze import run_report
from ..log import print_banner, setup_console
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
    "-f", "--fit_card", type=str, default=None, required=True, help="fit card name",
)

n_replica = click.option(
    "-n",
    "--n_replica",
    type=int,
    default=None,
    required=True,
    help="Number of the replica",
)

log_file = click.option(
    "-l",
    "--log_file",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    required=False,
    help="path to log file",
)


@base_command.command("NS")
@runcard_path
@fit_card
@log_file
def nested_sampling(runcard_path: pathlib.Path, fit_card: str, log_file: pathlib.Path):
    """Run a fit with |NS|.

    Parameters
    ----------
    runcard_path :
        path to runcard
    fit_card :
        fit runcard name
    log_file :
        path to log file
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        setup_console(log_file)
        print_banner()
        log.console.log("Running : Nested Sampling Fit ")
        runner = Runner.from_file(runcard_path, fit_card)
    else:
        runner = None

    runner = comm.bcast(runner, root=0)
    runner.run_analysis("NS")


@base_command.command("MC")
@runcard_path
@fit_card
@n_replica
@log_file
def monte_carlo_fit(
    runcard_path: pathlib.Path, fit_card: str, n_replica: int, log_file: pathlib.Path
):
    """Run a fit with |MC|.

    Parameters
    ----------
    runcard_path :
        path to runcard
    fit_card :
        fit runcard name
    n_replica :
        replica number
    log_file :
        path to log file
    """
    setup_console(log_file)
    print_banner()
    log.console.log("Running : MonteCarlo Fit")
    runner = Runner.from_file(runcard_path, fit_card, n_replica)
    runner.run_analysis("MC")


@base_command.command("PF")
@runcard_path
@fit_card
@click.option(
    "-n",
    "--n_replica",
    type=int,
    default=None,
    required=True,
    help="total number of replica to save",
)
@click.option(
    "-c",
    "--clean_rep",
    is_flag=True,
    default=False,
    help="remove the replica file",
)
def post_fit(
    runcard_path: pathlib.Path, fit_card: str, n_replica: int, clean_rep: bool
):
    """Run postfit selection over |MC| replicas.

    Parameters
    ----------
    runcard_path :
        path to runcard
    fit_card :
        fit runcard name
    n_replica :
        total number of replica to save
    clean_rep :
        if True revove replica folders
    """
    postfit = Postfit.from_file(runcard_path, fit_card)
    postfit.save(n_replica)
    if clean_rep:
        postfit.clean()


@base_command.command("SCAN")
@runcard_path
@fit_card
@click.option(
    "-n",
    "--n_replica",
    type=int,
    default=0,
    required=False,
    help="number of replicas used during the scan, default (0) will use only experiemental data",
)
@click.option(
    "-b",
    "--bounds",
    type=bool,
    is_flag=True,
    default=False,
    help="compute also the chi2 bounds",
)
def scan(runcard_path: pathlib.Path, fit_card: str, n_replica: int, bounds: bool):
    r"""Plot idividual :math:`\chi^2` profiles for
    all the free parameters.

    Parameters
    ----------
    runcard_path :
        path to runcard
    fit_card :
        fit runcard name
    n_replica :
        number of replica to use
    bounds :
        if True compute also the :math:`\chi^2` bounds
    """
    runner = Runner.from_file(runcard_path, fit_card)
    runner.chi2_scan(n_replica, bounds)


@base_command.command("R")
@runcard_path
@fit_card
def report(runcard_path: pathlib.Path, fit_card: str):
    """Run a fit report.

    Parameters
    ----------
    runcard_path :
        path to runcard
    fit_card :
        fit runcard name
    """
    run_report(runcard_path, fit_card)
