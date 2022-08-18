import torch

# Width of Regression Bins
WIDTH = 1/3

# Regression Bin Centers
CENTERS = [-3/2 * WIDTH, -1/2 * WIDTH, 1/2 * WIDTH, 3/2 * WIDTH]

# Regression Bin Boundaries
BOUNDARIES = [-WIDTH, 0, WIDTH]


def _unused_relabel(label):
    """Relabel the initial values provided to numerical 1-4

    Args:
        label (int): Value 1-4

    Returns:
        int, None: Relabeled values
    """
    if label == 1:
        return 2
    elif label == 2:
        return 4
    elif label == 3:
        return 3
    elif label == 4:
        return 1
    elif label is None:
        return None
    else:
        raise AssertionError(f"Shouldn't have label {label}")


def approx_eq(a, b):
    """Determines if two values are approximately equal

    Within 1e-4 tolerance on either side

    Args:
        a (float): value 1
        b (float): value 2

    Returns:
        bool: a == b within 1e-4 boundary
    """
    return b - 1e-4 <= a <= b + 1e-4


def relabel_for_regression(label):
    return CENTERS[int(label) - 1]


def relabel_for_regression_optimized(labels, label_max=4, label_min=1):
    """Relabels a set of labels in to the values prescribed by CENTERS

    Args:
        labels (array): Labels to be reformatted
        label_max (int, optional): Maximum Label. Defaults to 4.
        label_min (int, optional): Minimum label. Defaults to 1.

    Returns:
        array: Relabeled array of labels
    """
    norm_factor = (CENTERS[-1] - CENTERS[0]) / (label_max - label_min)
    labels *= norm_factor
    labels -= label_min * norm_factor - CENTERS[0]
    return labels


def regression_correctness(truth, pred):
    """Determines the correctness of a regression prediction on a single sample

    Correct is defined as the regression prediction falling into the bucket given by
    the ground truth.

    Args:
        truth (float): Ground truth (in CENTERS)
        pred (float): Regression prediction

    Returns:
        bool: True if the regression prediction was in the correct bucket
    """
    if pred <= BOUNDARIES[0]:
        return approx_eq(truth, CENTERS[0])
    elif pred >= BOUNDARIES[2]:
        return approx_eq(truth, CENTERS[3])
    elif BOUNDARIES[1] <= pred < BOUNDARIES[2]:
        return approx_eq(truth, CENTERS[2])
    elif BOUNDARIES[0] < pred < BOUNDARIES[1]:
        return approx_eq(truth, CENTERS[1])


def regression_label_preds(pred):
    """Takes a regression prediction and buckets it based on its value

    Potential values are 1, 2, 3, 4 based on prediction position relative
    to BOUNDARIES constant.

    Args:
        pred (float): Regression prediction

    Returns:
        int: Bucketed prediction in [1, 2, 3, 4]
    """
    if pred <= BOUNDARIES[0]:
        return 1
    elif pred >= BOUNDARIES[2]:
        return 4
    elif BOUNDARIES[1] <= pred < BOUNDARIES[2]:
        return 3
    elif BOUNDARIES[0] < pred < BOUNDARIES[1]:
        return 2


def humanize_pred(pred):
    """Take a regression prediction and make it readable

    Takes regression predictions and rescales them such that
    the CENTERS become 1, 2, 3, 4

    Args:
        pred (array or float): Regression prediction

    Returns:
        array or float: Rescaled predictions
    """
    return (pred + 5/2 * WIDTH) / WIDTH


def get_device():
    """Get torch device for model training 

    Returns:
        str: torch device descriptor
    """
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'
