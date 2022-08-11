# -*- coding: utf-8 -*-
import inspect
import logging

import rich.align
from rich.console import Console
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)

# TODO: eventually redirect the output to a log file
console = Console()


def print_banner(logger):
    banner = rich.align.Align(
        rich.panel.Panel.fit(
            inspect.cleandoc(
                r"""
                     ____  __  __ _____ _____ _ _____
                    / ___||  \/  | ____|  ___(_)_   _|
                    \___ \| |\/| |  _| | |_  | | | |
                     ___) | |  | | |___|  _| | | | |
                    |____/|_|  |_|_____|_|   |_| |_|

                A Standard Model Effective Field Theory Fitter
                """
            ),
            rich.box.SQUARE,
            padding=1,
            style="magenta",
        ),
        "center",
    )
    console.print(banner)
