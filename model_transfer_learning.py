import os
import pathlib
from datetime import datetime

import torch
import pandas as pd

from cli import validation_transfer_learning
from config_tools import seed_randomness
from supervised_modeling_dataset_tools import (
    SupervisedCellsDataset,
    train_supervised_model,
    load_supervised_from_unsupervised
)


if __name__ == '__main__':
    # Use CLI to parse inputs
    preds_path, metadata_path, train_index,\
    test_index, cell_images_path, cell_masks_path,\
    pretrained_model, hidden_units,\
    seed, n_epochs = validation_transfer_learning()
    rng = seed_randomness(seed)

    # Make directory to reproduce model results
    model_results_dir = os.path.join(preds_path,
                                     "SupervisedModelRun_" +
                                     datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"))

    # Create directory for model results
    if not os.path.exists(model_results_dir):
        os.mkdir(model_results_dir)
    else:
        raise AssertionError(
            f"Path exists: {model_results_dir}, not overwriting")

    # Add helper shell script for assistance in reproducing a particular run
    reproduce_run = os.path.join(model_results_dir, "reproduce_run.sh")
    active_path = pathlib.Path(__file__).parent.absolute()

    # Make predictions path absolute for better documentation
    if not os.path.isabs(preds_path):
        preds_path = os.path.join(active_path, preds_path)

    with open(reproduce_run, "w") as shell_script:
        shell_script.write(f"#! /bin/bash\npython3 " +
                           f"{os.path.join(active_path, 'model_transfer_learning.py')} " +
                           f"-s {seed} -p {preds_path} -e {n_epochs} " +
                           f"-im {cell_images_path} -ma {cell_masks_path} " +
                           f"-m {metadata_path} -tr {train_index} -te {test_index} "+
                           f"-pm {pretrained_model} -u {hidden_units}")

    metadata = pd.read_pickle(metadata_path)

    model = load_supervised_from_unsupervised(pretrained_model, hidden_units)
