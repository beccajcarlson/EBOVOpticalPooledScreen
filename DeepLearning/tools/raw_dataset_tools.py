import os
import pandas as pd
import numpy as np

from skimage import io
from tqdm import tqdm


def fmt_dir(plate, well, tile, prefix="./data_samples/", mask=False, centers=False):
    """Formats filepath for a particular cell

    Args:
        plate (str): Cell plate
        well (str): Cell well
        tile (int): Cell tile
        prefix (str, optional): Prefix to find cell images, masks, centers. Defaults to "./cells/".
        mask (bool, optional): Whether to return mask path. Defaults to False.
        centers (bool, optional): Whether to return centers path. Defaults to False.

    Returns:
        str: Filepath
    """
    if mask:
        return os.path.join(prefix,
                            f"GW{plate}/Well{well}_Tile-{str(int(tile)).zfill(4)}.cellpose.tif")
    elif centers:
        return os.path.join(prefix,
                            f"centers/GW{plate}/Well{well}_Tile-{str(int(tile)).zfill(4)}.centers.npy")
    else:
        return os.path.join(prefix,
                            f"GW{plate}/Well{well}_Tile-{str(int(tile)).zfill(4)}.phenotype_corr.tif")


def extract_centers(metadata_path="./normed_all.hdf", centers_folder="./centers/",
                    tile_column_name="tile_y",
                    image_size=1480, border_size=32):
    """ Extracts the centers of cells based on plate, well, tile, and saves them to a file

    Used for optimized preprocessing of dataset and training speedup

    Args:
        metadata_path (str, optional): Path to hdf file of metadata. Defaults to "./normed_all.hdf".
        centers_folder (str, optional): Path to generate folder containing centers. Defaults to "./centers/".
        tile_column_name (str, optional): Column name in Pandas DataFrame of tile. Defaults to "tile_y".
        image_size (int, optional): Dimension of square tile. Defaults to 1480.
        border_size (int, optional): Border size to use for clipping images. Defaults to 32.
    """
    metadata = pd.read_hdf(metadata_path)
    metadata = metadata.groupby(["plate", "well", tile_column_name])

    if not os.path.exists(centers_folder):
        os.mkdir(centers_folder)

    for ind, g in tqdm(metadata, total=len(metadata)):
        plate, well, tile = ind

        plate_path = os.path.join(centers_folder, f"GW{plate}")
        if not os.path.exists(plate_path):
            os.mkdir(plate_path)

        # Path for the centers of a particular cell
        path = os.path.join(
            plate_path, f"Well{well}_Tile-{str(int(tile)).zfill(4)}.centers.npy")
        centers = g[["i_og_0", "j_og_0"]].to_numpy()
        centers = centers.round().astype(int)

        # Remove cells whose border interferes with the boundary
        centers = np.delete(centers, np.where(
            (centers < border_size) | (centers > image_size-border_size))[0], axis=0)

        np.save(path, centers)


def extract_image_subset(subset, border=64, channels=6,
                         apply_mask=True, save_images=None, save_masks=None):
    """Given a small subset of cells, extract their images

    Args:
        subset (pandas DataFrame): Subset of cells to extract
        border (int, optional): Border size of image. Defaults to 64.
        channels (int, optional): Number of channels. Defaults to 6.
        apply_mask (bool, optional): Whether to apply the mask to the image. Defaults to True.
        save_images (str, optional): The path at which to save images. Defaults to None.
        save_masks (str, optional): The path at which to save masks. Defaults to None.

    Returns:
        array, array: image and mask numpy arrays
    """

    # Allocate arrays for images and masks
    temp_images = np.zeros((subset.shape[0], channels, border, border))
    temp_masks = np.zeros((subset.shape[0], channels, border, border))

    # Border relative to center
    center_border = border // 2

    pbar = tqdm(enumerate(subset.iterrows()), total=len(subset))
    for j, (_, row) in pbar:

        # Extract image, min/max normalize across channels
        image_path = fmt_dir(row.plate, row.well, row.tile)
        image = io.imread(image_path)
        image /= np.max(image, axis=(1, 2))[:, np.newaxis, np.newaxis]

        channel_dim, x_dim, y_dim = image.shape

        # Extract mask
        mask_path = fmt_dir(row.plate, row.well, row.tile, mask=True)
        mask = io.imread(mask_path)

        # Determine center coordinate and bounding box
        x, y = round(row.i_og_0), round(row.j_og_0)
        x_min, x_max, y_min, y_max = x-center_border, x + \
            center_border, y-center_border, y+center_border

        # Repeat mask over channels
        mask_region = np.where(mask == mask[x, y], 1, 0)
        mask_region = np.repeat(
            mask_region[np.newaxis, ...], channel_dim, axis=0)

        # If the bounding box centered at the desired coordinate would pass the edge
        # inform user, and crop without centering
        if x_min < 0:
            print(f"Could not center crop, {(row.plate, row.well, row.tile)}")
            x_min = 0
            x_max = border
        elif x_max > x_dim:
            print(f"Could not center crop, {(row.plate, row.well, row.tile)}")
            x_max = x_dim
            x_min = x_dim-border

        if y_min < 0:
            print(f"Could not center crop, {(row.plate, row.well, row.tile)}")
            y_min = 0
            y_max = border
        elif y_max > y_dim:
            print(f"Could not center crop, {(row.plate, row.well, row.tile)}")
            y_max = y_dim
            y_min = y_dim-border

        # Extract bounding box for mask
        mask_slice = mask_region[..., x_min: x_max, y_min: y_max]

        if apply_mask:
            cell_slice = image[..., x_min: x_max, y_min: y_max] * mask_slice
        else:
            cell_slice = image[..., x_min: x_max, y_min: y_max]

        temp_images[j] = cell_slice
        temp_masks[j] = mask_slice

    if isinstance(save_images, str):
        np.save(save_images, temp_images)

    if isinstance(save_masks, str):
        np.save(save_masks, temp_masks)

    return temp_images, temp_masks
