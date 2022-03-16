# -*- coding: utf-8 -*-
import argparse
import pathlib

import smefit

path = pathlib.Path(__file__).absolute().parent 

if __name__ == "__main__":

    # Mini argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        default=None,
        type=str,
        help="mode: NS (Nested Sampling)",
    )
    parser.add_argument(
        "-f",
        "--fit_cards",
        nargs="+",
        required=True,
        default=None,
        type=str,
        help="fit card name/s, to fit pass one card only",
    )
    parser.add_argument(
        "-p",
        "--runcard_path",
        required=False,
        default=path / 'runcards',
        type=str,
        help="path to runcard",
    )

    args = parser.parse_args()

    smefit.run(args.runcard_path, args.mode, args.fit_cards[0])
