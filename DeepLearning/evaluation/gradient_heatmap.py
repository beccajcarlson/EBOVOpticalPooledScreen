import sys
import pathlib
sys.path.append(pathlib.Path(__file__).parents[1].__str__())

import numpy as np
from captum.attr import DeepLift

from tools.image_tools import cell_image
from tools.config_tools import get_device
from modeling.model import ConvAutoencoderWithHead
from tools.processed_dataset_tools import PHENOTYPES


def generate_gradient_heatmap(image_tensor, model_path,
                              hidden_dims=256,
                              channels=(0, 1, 2, 3, 4, 5),
                              channel_names=("DAPI", "FISH", "VP35",
                                             "Jun", "Vimentin", "LAMP1"),
                              target_label="Faint"):
    """Generates a gradient-per-pixel heatmap for given input tensor and model

    Args:
        image_tensor (Torch tensor): Image with channels as first axis
        model_path (str): Path to pretrained autoencoder model with head
        hidden_dims (int, optional): Number of hidden units in classification head.
            Defaults to 256.
        channels (tuple, optional): Channels to show. Defaults to (0, 1, 2, 3, 4, 5).
        channel_names (tuple, optional): Channel names, in order.
            Defaults to ("DAPI", "FISH", "VP35", "Jun", "Vimentin", "LAMP1").
        target_label (str, optional): Target label for input image.
            Defaults to "Faint".

    Returns:
        numpy array, Matplotlib figure, numpy array, numpy array:
            Gradient per pixel in the initial input image;
            Visualization of gradient-per-pixel in initial input image;
            Reconstructed image from autoencoder;
            Log-Softmax output with per-phenotype values
    """
    assert target_label in PHENOTYPES,\
        f"Must choose a target label in the set {PHENOTYPES}, got {target_label}"
    target = PHENOTYPES.index(target_label)
    device = get_device()

    # Instantiate model on device, add batch dimension to image
    model = ConvAutoencoderWithHead(hidden=hidden_dims, only_phenotype=True)
    model.load_state_dict(model_path, map_location=device)
    model.to(device)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Get gradient-per-pixel measurement
    gradient_cam = DeepLift(model)
    gradient = gradient_cam.attribute(image_tensor, target=target)\
                           .squeeze().detach().cpu().numpy()

    heatmap = cell_image(gradient, channels, channel_names,
                         norm_0=True, grad_img=True)

    reconstructed, label, _ = model._forward_return_all(image_tensor)
    reconstructed = reconstructed.squeeze().detach().cpu().numpy()
    label = label.squeeze().detach().cpu().numpy()

    return gradient, heatmap, reconstructed, label
