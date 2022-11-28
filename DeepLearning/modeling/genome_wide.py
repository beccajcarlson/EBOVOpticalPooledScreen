import os
import sys
import pathlib
sys.path.append(pathlib.Path(__file__).parents[1].__str__())

from datetime import datetime

import torch
import pandas as pd

from tools.cli import validation
from tools.config_tools import seed_randomness
from modeling.model import ConvAutoencoder
from tools.unsupervised_modeling_dataset_tools import (
    UnsupervisedCellsDataset, train_unsupervised_model, split_dataset
)


if __name__ == '__main__':
    # Use CLI to parse inputs
    preds_path, metadata_path, seed, n_epochs,\
        lr, test_size, stratify_by_plate = validation()
    rng = seed_randomness(seed)

    # Make directory to reproduce model results
    model_results_dir = os.path.join(preds_path,
                                     "UnsupervisedModelRun_" +
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
        stratify_flag = '-n' if (not stratify_by_plate) else ''
        shell_script.write(f"#! /bin/bash\npython3 " +
                           f"{os.path.join(active_path, 'genome_wide.py')} " +
                           f"-s {seed} -p {preds_path} -e {n_epochs} " +
                           f"-l {lr} -m {metadata_path} -t {test_size} " +
                           f"{stratify_flag}")

    metadata = pd.read_pickle(metadata_path)
    model = ConvAutoencoder()

    train_ds, test_ds = split_dataset(metadata, seed=seed,
                                      test_size=test_size,
                                      stratify_by_plate=stratify_by_plate)

    train_dl = UnsupervisedCellsDataset(train_ds, rng, test=False,
                                        cell_size=64, batch_size=256)
    test_dl = UnsupervisedCellsDataset(test_ds, rng, test=True, cell_size=64)
    losses, trained_model = train_unsupervised_model(train_dl, test_dl,
                                                     model, n_epochs, lr)

    trained_model_path = os.path.join(model_results_dir,
                                      "unsupervised_trained_model.pth")
    torch.save(trained_model.state_dict(), trained_model_path)

    # Generate DataFrame of losses
    loss_df = pd.DataFrame(losses, columns=["Train Loss", "Test Loss"])
    loss_df = loss_df.reset_index().rename(columns={"index": "Epoch"})
    loss_df["Epoch"] += 1  # Reindex from 0,... to 1,...

    loss_df_path = os.path.join(model_results_dir, "losses.pkl")
    loss_df.to_pickle(loss_df_path)
