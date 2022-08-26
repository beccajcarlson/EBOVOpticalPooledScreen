import pandas as pd
import numpy as np
import torch
import pickle

from skimage import io
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from raw_dataset_tools import fmt_dir
from modeling_tools import get_device, humanize_pred


class CellsDataset(Dataset):
    """Dataset for cell image extraction from raw files
    """

    def __init__(self, metadata, embedding_col="embedding"):
        """Initialize dataset

        metadata should contain each triple of ["plate", "well", "tile"] exactly once,
        as it extracts the cell images for each tile all at once. One way to generate
        this dataframe from a dataframe of all cells, raw_data, is:

        raw_data.groupby(["plate", "well", "tile"]).agg("first").reset_index()

        ** WILL ONLY WORK FOR 64 X 64 CELL IMAGES CURRENTLY **

        Args:
            metadata (pandas DataFrame): Dataframe containing columns ["plate", "well", "tile", "embedding"]
            embedding_col (str, optional): Column name in df denoting embedding index. Defaults to embedding.
        """
        self.indices = metadata
        self.embedding_col = embedding_col

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Extract row in df corresponding to tile
        sample_idx = self.indices.iloc[idx]

        # Extract mask, image, and centers filepaths
        mask_filename = fmt_dir(sample_idx.plate, sample_idx.well,
                                sample_idx.tile, mask=True)
        img_filename = fmt_dir(sample_idx.plate, sample_idx.well,
                               sample_idx.tile, mask=False)
        centers_filename = fmt_dir(sample_idx.plate, sample_idx.well,
                                   sample_idx.tile, centers=True)

        # Load centers, images, masks
        sample_cells = np.load(centers_filename)

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

        # Return cells, embedding value of sample
        return temp_images, sample_idx[self.embedding_col]


def extract_embeddings_and_labels(model, classifier=True,
                                  df_path="./metadata_embeddings_V2.pkl"):
    """For a particular model and dataframe of cells, extract latent space embeddings

    Can also extract classifications.

    Args:
        model (ConvAutoencoderWithHead): Model to generate embeddings with
        classifier (bool, optional): Whether autoencoder head is classifier or regressor. Defaults to True.
        df_path (str, optional): Path to dataframe of cells. Defaults to "./metadata_embeddings_V2.pkl".

    Returns:
        (numpy array, numpy array): latent space embeddings, autoencoder head labels [1-4]
    """

    # Extract raw data from path, round centers
    raw_data = pd.read_pickle(df_path)
    raw_data["i_og_0"] = raw_data["i_og_0"].round()
    raw_data["j_og_0"] = raw_data["j_og_0"].round()

    # Restrict centers to those which fall within the crop boundary [32] from the border
    raw_data = raw_data[(raw_data["i_og_0"] >= 32) & (raw_data["j_og_0"] >= 32) &
                        (raw_data["i_og_0"] <= (1480-32)) & (raw_data["j_og_0"] <= (1480-32))].copy()
    raw_data["embedding_index"] = np.nan

    # Sort cells in a stable fashion, extract index
    sorted_indices = raw_data.sort_values(
        ["plate", "well", "tile"], kind="stable").index

    # Set embedding index such that all cells of the same tile have continuous indices
    raw_data.loc[sorted_indices, "embedding_index"] = np.arange(
        len(raw_data), dtype=np.uint64)

    # Group data by (plate, well, tile) and initialize data storage arrays
    # WARNING: embeddings_npy can be extremely large (40M cells ~ 150 GB array)

    # If the above is an issue, consider running the function multiple times
    # on smaller DataFrames
    metadata = raw_data.groupby(["plate", "well", "tile"])
    embeddings_npy = np.zeros((len(raw_data), 2048), dtype=np.float16)
    labels_npy = np.zeros(len(raw_data), dtype=np.int8)

    device = get_device()

    def vrange(starts, stops):
        """Create concatenated ranges of integers for multiple start/stop

        SOURCE: https://codereview.stackexchange.com/a/84980

        Parameters:
            starts (1-D array_like): starts for each range
            stops (1-D array_like): stops for each range (same shape as starts)

        Returns:
            numpy.ndarray: concatenated ranges

        For example:

            >>> starts = [1, 3, 4, 6]
            >>> stops  = [1, 5, 7, 6]
            >>> vrange(starts, stops)
            array([3, 4, 4, 5, 6])

        """
        stops = np.array(stops)
        l = stops - starts  # Lengths of each range.
        return np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())

    # Extract starting embedding indices from metadata
    all_sample_indices = metadata.agg("first").reset_index()[
        ["plate", "well", "tile", "embedding_index"]]
    model.eval()

    # Initialize cell dataset
    dataset = CellsDataset(all_sample_indices, "embedding_index")

    # Collator concatenates all tile images and generates ranges of embedding indices for each tile
    def collator(x): return (torch.cat([torch.Tensor(ims) for ims, _ in x], dim=0).float(),
                             vrange([int(starts) for _, starts in x], [int(starts)+len(ims) for ims, starts in x]))

    # Dataloader responsible for generating images in a parallelized manner
    dataloader = DataLoader(dataset, batch_size=24,
                            collate_fn=collator, num_workers=24)

    # Iterate over dataloader
    for _, data_batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        images, embeddings_idx = data_batch
        images = images.to(device)

        # Predict on whole batch
        with torch.no_grad():
            _, labels, embeds = model(images)

        embeds = embeds.detach().cpu().numpy()

        # Labels are extracted differently depending on whether head is
        # classifier or regressor
        if classifier:
            labels = labels.argmax(axis=1).detach(
            ).cpu().numpy().ravel().astype(np.int8) + 1
        else:
            labels = humanize_pred(
                labels.detach().cpu().numpy().ravel()).astype(np.float16)

        # Place embeddings and labels at correct indices in array
        embeddings_npy[embeddings_idx] = embeds.astype(np.float16)
        labels_npy[embeddings_idx] = labels

    return embeddings_npy, labels_npy


def load_svm_classifier(path):
    """Loads an SVM classifier from a path

    Args:
        path (str): Path to classifier

    Returns:
        sklearn SVM model: classifier
    """
    with open(path, 'rb') as f:
        classifier = pickle.load(f)

    return classifier


def predict_svm(svm_classifier, feature_path, per_split=None):
    """Performs prediction using SVM classifier on features

    Can perform incremental prediction if needed for RAM constraints

    Args:
        svm_classifier (sklearn SVM): SVM classifier
        feature_path (str): Path to numpy array containing features [n samples x d features]
        per_split (int, optional): Number of samples per prediction split, if desired. Defaults to None.

    Returns:
        array: vector of SVM predictions for each sample
    """
    # Load features lazily, as needed
    features = np.load(feature_path, mmap_mode="r")

    # Store incremental predictions and most recent split
    preds_ = []
    prev_split = 0

    if per_split is not None:
        # If desired to split training into chunks, iterate over each chunk, predicting
        splits = per_split * \
            np.arange(1, len(features) // per_split).astype(int)
        for split in tqdm(splits):
            chunk = svm_classifier.predict(features[prev_split:split])
            preds_.append(chunk)
            prev_split = split

    # Accumulate final chunk
    chunk = svm_classifier.predict(features[prev_split:])
    preds_.append(chunk)

    # Concatenate all chunks and return predictions
    return np.concatenate(preds_).astype(np.int8)
