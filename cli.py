import os
import argparse
import pathlib


def cli():
    active_path = pathlib.Path(__file__).parent.absolute()
    opts = argparse.ArgumentParser()
    opts.add_argument('-p', dest='preds_path', default=os.path.join(active_path.parent, "models"),
                      type=str, help=f'Path to save model - Default {os.path.join(active_path.parent, "models")}')
    opts.add_argument('-s', dest='seed', default=5, type=int,
                      help='Seed of random generators - Default 5')
    return opts.parse_args()


def validation():
    args = cli()

    if not os.path.exists(args.preds_path):
        os.mkdir(args.preds_path)

    return args.preds_path, int(args.seed)
