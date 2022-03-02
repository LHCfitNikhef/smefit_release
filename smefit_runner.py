import pathlib
import argparse
#import warnings

from smefit import RUNNER


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

    args = parser.parse_args()

    runner = RUNNER(path)

    
    if args.mode == "NS":
        # run NS
        runner.ns(args.fit_cards[0])
    else:
        raise NotImplementedError(
            f"MODE={args.mode} is not valid, the only implemented feature atm is NS"
        )
