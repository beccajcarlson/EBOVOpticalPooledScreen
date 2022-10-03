import torch
import numpy as np
from tqdm import tqdm
from skimage import io

from torch.utils.data import DataLoader, Dataset
from config_tools import get_device
from raw_dataset_tools import fmt_dir
from model import my_custom_mse


class UnsupervisedCellsDataset(Dataset):
    """Dataset for unsupervised cell image extraction from raw files
    """

    def __init__(self, metadata, rng=None, test=False, batch_size=256):
        """Initialize dataset

        metadata should contain each triple of ["plate", "well", "tile"] exactly once,
        as it extracts the cell images for each tile all at once. One way to generate
        this dataframe from a dataframe of all cells, raw_data, is:

        raw_data.groupby(["plate", "well", "tile"]).agg("first").reset_index()

        ** WILL ONLY WORK FOR 64 X 64 CELL IMAGES CURRENTLY **

        Args:
            metadata (pandas DataFrame): Dataframe containing columns ["plate", "well", "tile"]
            rng (np.random.Generator, optional): Random number generator or None. Defaults to None.
            test (bool, optional): Whether metadata is test (True) or train (False). Defaults to False.
            batch_size (int, optional): Maximum size of each training batch. Defaults to 256.
        """
        self.batch_size = batch_size
        self.metadata = metadata
        self.test = test

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Extract row in df corresponding to tile
        sample_idx = self.metadata.iloc[idx]

        # Extract mask, image, and centers filepaths
        mask_filename = fmt_dir(sample_idx.plate, sample_idx.well,
                                sample_idx.tile, mask=True)
        img_filename = fmt_dir(sample_idx.plate, sample_idx.well,
                               sample_idx.tile, mask=False)
        centers_filename = fmt_dir(sample_idx.plate, sample_idx.well,
                                   sample_idx.tile, centers=True)

        # Load centers, images, masks
        sample_cells = np.load(centers_filename)

        # If not a test dataset, select a random sample of cells as the batch
        if not self.test:
            random_samples = self.rng.choice(sample_cells.shape[0],
                                              min(sample_cells.shape[0],
                                                  self.batch_size),
                                              replace=False)
            sample_cells = sample_cells[random_samples]

        # Min-Max scale images across channels
        image = io.imread(img_filename).astype(np.float32)
        image /= np.max(image, axis=(1, 2))[:, np.newaxis, np.newaxis]
        mask = io.imread(mask_filename).astype(np.float32)
        mask = np.repeat(mask[np.newaxis, ...], 6, axis=0)

        # Reindex centers to work for numpy stride tricks
        ind_x = sample_cells - 32

        # Stride across image and mask, extracting views of correct size
        multi_slice_image = np.lib.stride_tricks.sliding_window_view(
            image, (6, 64, 64))
        multi_slice_mask = np.lib.stride_tricks.sliding_window_view(
            mask, (6, 64, 64))

        # Select desired views in mask and image strides
        # Make masks boolean for each image [vectorized]
        temp_masks = multi_slice_mask[0, ind_x[:, 0],
                                      ind_x[:, 1], ...].astype(np.float32)
        temp_masks = np.where(
            temp_masks == temp_masks[..., 31:32, 31:32], 1, 0)
        temp_images = multi_slice_image[0,
                                        ind_x[:, 0], ind_x[:, 1], ...] * temp_masks

        # Cast to torch Tensors and return images and masks
        temp_images, temp_masks = torch.Tensor(
            temp_images), torch.Tensor(temp_masks)
        return temp_images, temp_masks


def generate_train_test_dataloaders(train_dataset, test_dataset):
    """Generates training and testing DataLoaders from UnsupervisedCellsDataset

    Args:
        train_dataset (UnsupervisedCellsDataset): Training cells
        test_dataset (UnsupervisedCellsDataset): Testing cells

    Returns:
        DataLoader, DataLoader: Train and Test dataloaders
    """

    # Concatenate images and masks along first axis
    def collator(x): return (torch.cat([ims for ims, _ in x], dim=0).float(),
                             torch.cat([masks for _, masks in x], dim=0).bool())

    # Prepare dataloaders
    # OPTIMIZED FOR USE WITH 24 GB OF GPU SPACE
    train_dl = DataLoader(train_dataset, batch_size=32,
                          collate_fn=collator, num_workers=32)
    test_dl = DataLoader(test_dataset, batch_size=16,
                         collate_fn=collator, num_workers=16)

    return train_dl, test_dl


def train_unsupervised_model(train_dl, test_dl, model, n_epochs=10, lr=0.001):
    """Trains an unsupervised autoencoder

    Uses Adam optimizer and custom MSE only penalizing errors within the mask

    Args:
        train_dl (DataLoader): Training dataloader
        test_dl (DataLoader): Testing dataloader
        model (ConvAutoencoder): Unsupervised model to train
        n_epochs (int, optional): Number of epochs to train for. Defaults to 10.
        lr (float, optional): Model learning rate. Defaults to 0.001.

    Returns:
        list((float, float)), ConvAutoencoder: Train, Test losses for each epoch and trained model
    """

    # Get device to use
    device = get_device()

    # Loss function
    criterion1 = my_custom_mse

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Overall Stats
    losses = []

    pbar = tqdm(range(1, n_epochs+1))
    for epoch in pbar:
        # Monitor training loss
        train_loss1 = 0.0
        test_loss1 = 0.0

        # Training
        for _, data_batch in tqdm(enumerate(train_dl), total=len(train_dl)):
            images, masks = data_batch
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs, _ = model(images)

            loss1 = criterion1(outputs * masks,
                               images * masks, masks)
            loss1.backward()
            optimizer.step()
            train_loss1 += loss1.item()

        model.eval()
        with torch.no_grad():
            # Testing
            for _, test_data in tqdm(enumerate(test_dl), total=len(test_dl)):
                test_images, test_masks = test_data
                test_images, test_masks = test_images.to(device),\
                    test_masks.to(device)
                test_out, _ = model(test_images)
                test_loss = criterion1(test_out * test_masks,
                                       test_images * test_masks,
                                       test_masks)
                test_loss1 += test_loss.item()
        model.train()

        train_loss1 = train_loss1 / len(train_dl)
        test_loss1 = test_loss1 / len(test_dl)
        losses.append((train_loss1, test_loss1))

        pbar.set_description('Epoch: {} \tTraining Loss 1: {:.6f}\tTest Loss 1: {:.6f}'
                             .format(epoch, train_loss1, test_loss1))

    return losses, model
