import os
import argparse
import pathlib


def cli_genome_wide():
    active_path = pathlib.Path(__file__).parent.absolute()
    opts = argparse.ArgumentParser()
    opts.add_argument('-s', dest='seed', default=5, type=int,
                      help='Seed of random generators - Default 5')
    opts.add_argument('-e', dest='epochs', default=5, type=int,
                      help='Number of epochs to train for - Default 5')
    opts.add_argument('-l', dest='learning_rate', default=0.001, type=float,
                      help='Learning rate - Default 0.001')
    opts.add_argument('-p', dest='preds_path', default=os.path.join(active_path.parent, "models"),
                      type=str, help=f'Path to save model - Default {os.path.join(active_path.parent, "models")}')
    opts.add_argument('-t', dest='test_size', default=0.2, type=float,
                      help='Test set size in (0, 1) - Default 0.2')
    opts.add_argument('-n', dest='stratify_by_plate', default=True, action='store_false',
                      help='Set this flag to NOT stratify train and test sets by plate - Default unset')
    opts.add_argument('-m', dest='metadata_path', default="./data_samples/metadata.pkl",
                      type=str, help='Path to find cell metadata - Default ./data_samples/metadata.pkl')
    return opts.parse_args()


def cli_transfer_learning():
    active_path = pathlib.Path(__file__).parent.absolute()
    opts = argparse.ArgumentParser()
    opts.add_argument('-s', dest='seed', default=5, type=int,
                      help='Seed of random generators - Default 5')
    opts.add_argument('-e', dest='epochs', default=100, type=int,
                      help='Number of epochs to train for - Default 100')
    opts.add_argument('-pm', dest='pretrained_model', default="model.pth", type=str,
                      help='Path to pretrained model - Default model.pth')
    opts.add_argument('-u', dest='hidden_units', default=256, type=int,
                      help='Number of hidden units - Default 256')
    opts.add_argument('-tr', dest='train_index', default='train_index.pkl', type=str,
                      help='Path to find training indices - Default train_index.pkl')
    opts.add_argument('-te', dest='test_index', default='test_index.pkl', type=str,
                      help='Path to find test indices - Default test_index.pkl')
    opts.add_argument('-p', dest='preds_path', default=os.path.join(active_path.parent, "models"),
                      type=str, help=f'Path to save trained model - Default {os.path.join(active_path.parent, "models")}')
    opts.add_argument('-m', dest='metadata_path', default="./data_samples/metadata.pkl",
                      type=str, help='Path to find cell metadata - Default ./data_samples/metadata.pkl')
    opts.add_argument('-im', dest='cell_images_path', default="images.npy",
                      type=str, help='Path to find cell images - Default images.npy')
    opts.add_argument('-ma', dest='cell_masks_path', default="masks.npy",
                      type=str, help='Path to find cell masks - Default masks.npy')
    return opts.parse_args()


def validation():
    args = cli_genome_wide()

    if not os.path.exists(args.preds_path):
        os.mkdir(args.preds_path)

    return args.preds_path, args.metadata_path,\
        int(args.seed), int(args.epochs), float(args.learning_rate),\
        float(args.test_size), bool(args.stratify_by_plate)


def validation_transfer_learning():
    args = cli_transfer_learning()

    if not os.path.exists(args.preds_path):
        os.mkdir(args.preds_path)

    return args.preds_path, args.metadata_path, args.train_index,\
        args.test_index, args.cell_images_path, args.cell_masks_path,\
        args.pretrained_model, int(args.hidden_units),\
        int(args.seed), int(args.epochs)
