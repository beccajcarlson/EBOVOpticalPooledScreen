import os
import argparse
import pathlib


def cli():
    active_path = pathlib.Path(__file__).parent.absolute()
    opts = argparse.ArgumentParser()
    opts.add_argument('-s', dest='seed', default=5, type=int,
                      help='Seed of random generators - Default 5')
    opts.add_argument('-e', dest='epochs', default=5, type=int,
                      help='Number of epochs to train for - Default 10')
    opts.add_argument('-l', dest='learning_rate', default=0.001, type=float,
                      help='Learning rate - Default 0.001')
    opts.add_argument('-p', dest='preds_path', default=os.path.join(active_path.parent, "models"),
                      type=str, help=f'Path to save model - Default {os.path.join(active_path.parent, "models")}')
    opts.add_argument('-m', dest='metadata_path', default="metadata.pkl",
                      type=str, help='Path to find cell metadata - Default metadata.pkl')
    return opts.parse_args()


def validation():
    args = cli()

    if not os.path.exists(args.preds_path):
        os.mkdir(args.preds_path)

    return args.preds_path, args.metadata_path,\
           int(args.seed), int(args.epochs), float(args.learning_rate)
