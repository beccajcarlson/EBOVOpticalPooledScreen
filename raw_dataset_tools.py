import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def fmt_dir(plate, well, tile, prefix="./cells/", mask=False, centers=False):
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
