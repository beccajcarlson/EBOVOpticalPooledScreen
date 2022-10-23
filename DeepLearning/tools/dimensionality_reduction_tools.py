import umap
import numpy as np

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from scipy.spatial import cKDTree


def extract_features(features, dimensionality_reducer="UMAP",
                     n_dim_out=2, standard_scale=False):
    """Extracts low-dimension features via dimensionality reduction

    Args:
        features (array): Set of input features (latent space representation) [n samples x k dims]
        dimensionality_reducer (str, optional): Selected technique. Defaults to "UMAP".
        n_dim_out (int, optional): Number of output dimensions. Defaults to 2.
        standard_scale (bool, optional): Whether to standard scale inputs. Defaults to False.

    Returns:
        array: Dimensionality-reduced features, shape [n x n_dim_out]
    """

    assert dimensionality_reducer.upper(
    ) in {"UMAP", "PCA"}, "Must select one of UMAP or PCA"

    if dimensionality_reducer == "UMAP":
        fit = umap.UMAP(n_components=n_dim_out)
    else:
        fit = PCA(n_components=n_dim_out)

    if standard_scale:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

    dim_red_features = fit.fit_transform(features)

    return dim_red_features


def find_nearest_points(point_set, target, n_points=1):
    """Finds nearest points to a target in a point set

    Args:
        point_set (array): Set of points to search over
        target (array): Center vector to find neighbors of
        n_points (int, optional): Number of nearest points. Defaults to 1.

    Returns:
        array, array: indices and corresponding nearest points
    """
    tree = cKDTree(point_set)
    q = tree.query(target, n_points)
    return q[-1], point_set[q[-1]]


def get_equidistant_points(p1, p2, total_pts):
    """Gets equidistant points between two anchor points

    Adapted from: https://stackoverflow.com/a/47443126

    Args:
        p1 (array): Anchor point 1
        p2 (array): Anchor point 2
        total_pts (int): Total number of equidistant points including endpoints

    Returns:
        array: Matrix of equidistant points, including endpoints
    """
    return np.array(list(zip(*[np.linspace(p1[i], p2[i], total_pts) for i in range(len(p1))])))


def perform_incremental_pca(data_path, n_components=3, per_split=100000):
    # Load dataset lazily, as needed
    dataset = np.load(data_path, mmap_mode='r')
    sklearn_pca = IncrementalPCA(n_components=n_components)

    # Store PCA dimension-reduced features
    pca_features = None

    # Determine splits to use for PCA
    splits = per_split * \
        np.arange(1, len(dataset) // per_split).astype(int)

    # Perform incremental partial-fits on chunks of the data
    prev_split = 0
    for split in tqdm(splits):
        chunk = dataset[prev_split:split]
        sklearn_pca.partial_fit(chunk)

        prev_split = split

    chunk = dataset[prev_split:]
    sklearn_pca.partial_fit(chunk)

    # Perform incremental dimensionality-reduction on chunks of the data
    prev_split = 0
    for split in tqdm(splits):
        chunk = dataset[prev_split:split]
        pca_chunk = sklearn_pca.transform(chunk)

        if pca_features is None:
            pca_features = pca_chunk
        else:
            pca_features = np.vstack((pca_features, pca_chunk))

        prev_split = split

    # Perform dimensionality reduction on the final chunk
    chunk = dataset[prev_split:]
    pca_chunk = sklearn_pca.transform(chunk)

    if pca_features is None:
        pca_features = pca_chunk
    else:
        pca_features = np.vstack((pca_features, pca_chunk))

    return pca_features
