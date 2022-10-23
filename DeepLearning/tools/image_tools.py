import numpy as np
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def cell_image(image_tensor, channels=[0, 1, 2, 3, 4, 5],
               channel_names=["DAPI", "FISH", "VP35",
                              "Jun", "Vimentin", "LAMP1"],
               cmap='viridis', norm_0=False, grad_img=False,
               min_=None, max_=None):
    """Generates a figure representing the selected channels of an input image
    The figure shows the channels in the order provided in the input list.

    Args:
        image_tensor (Torch tensor or Numpy array): Image with channels as first axis
        channels (list, optional): Channels to show. Defaults to [0, 1, 2, 3, 4, 5].
        channel_names (list, optional): Channel names, in order. Defaults to ["DAPI", "FISH", "VP35", "Jun", "Vimentin", "LAMP1"].
        cmap (str, optional): Choice of colormap. Defaults to 'viridis'.
        norm_0 (bool, optional): Normalize colors to have center 0. Defaults to False.
        grad_img (bool, optional): Image contains gradients per pixel. Defaults to False.
        min_ (float, optional): Minimum pixel value per image. Defaults to None.
        max_ (float, optional): Maximum pixel value per image. Defaults to None.

    Returns:
        Matplotlib figure of channels and names
    """
    assert len(channels) == len(
        channel_names), "Channel and Names lists do not match in size"

    fig, ax = plt.subplots(ncols=len(channels), figsize=(5 * len(channels), 5))

    if len(channels) == 1:
        ax = [ax]

    # Iterate over channels, populating each pane of the image
    for i, chan in enumerate(channels):
        if norm_0:
            # If gradient image, min/max normalize across all channels
            if grad_img:
                min_c = image_tensor.min()
                max_c = image_tensor.max()
                divnorm = colors.TwoSlopeNorm(vcenter=0, vmin=min_c, vmax=max_c)
            else:
                divnorm = colors.TwoSlopeNorm(vcenter=0)

            im = ax[i].imshow(image_tensor[chan, :, :],
                              cmap=cmap, norm=divnorm,
                              vmin=min_, vmax=max_)
        else:
            im = ax[i].imshow(image_tensor[chan, :, :], cmap=cmap,
                              vmin=min_, vmax=max_)

        ax[i].grid(False)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(channel_names[i], size=20)

        # If gradient image, only show colorbar on final image (same colorbar for all)
        if (grad_img and (i == len(channels) - 1)) or not grad_img:
            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.ax.tick_params(labelsize=18)

    plt.show()
    return fig


class Rotate:
    """ Rotates a cell image and its mask by a random angle 0 - 359 degrees

    Torch custom image transform.
    """

    def __init__(self, angle=None):
        """
        Args:
            angle (int, optional): Specific, fixed angle to rotate. Defaults to None.
        """
        self.angle = angle

    def __call__(self, x, mask):
        if self.angle is None:
            angle = np.random.uniform(-180, 180)
        else:
            angle = self.angle

        return TF.rotate(x, angle), TF.rotate(mask, angle)


class Flip:
    """ Flip a cell image and its mask over an axis

    Torch custom image transform.
    """

    def __init__(self, orientation="v", p=1):
        """
        Args:
            orientation (str, optional): Axis to flip, [v]ertical or [h]oriztonal. Defaults to "v".
            p (int, optional): Probability of flipping the image, versus leaving it as is. Defaults to 1.
        """
        assert orientation in [
            "v", "h"], "Must pick an orientation [v]ertical or [h]oriztonal"
        self.orientation = orientation
        self.p = p

    def __call__(self, x, mask):
        to_rotate = np.random.binomial(1, self.p)

        if to_rotate:
            return TF.vflip(x) if self.orientation == "v" else TF.hflip(x),\
                TF.vflip(mask) if self.orientation == "v" else TF.hflip(mask)

        else:
            return x.clone(), mask.clone()


class Gauss:
    """ Apply Gaussian kernel to a cell image and its mask

    Torch custom image transform.
    """

    def __init__(self, kernel=5, sigma=(0.2, 1.0)):
        """
        Args:
            kernel (int, optional): Kernel dimension. Defaults to 5.
            sigma (tuple, optional): Range of variances, chosen over uniformly. Defaults to (0.2, 1.0).
        """
        self.kernel = kernel
        self.sigma = sigma

    def __call__(self, x, mask):
        sigma_0 = np.random.uniform(*self.sigma)

        # Mask not blurred
        return TF.gaussian_blur(x, self.kernel, sigma_0), mask.clone()


class Affine:
    """ Apply Affine transform to a cell image and its mask

    Torch custom image transform.
    """

    def __init__(self, translate=5):
        """
        Args:
            translate (int, optional): Maximum intensity of Affine transform. Defaults to 5.
        """
        self.translate = translate

    def __call__(self, x, mask):
        translate_x = np.random.uniform(-self.translate,
                                        self.translate)
        translate_y = np.random.uniform(-self.translate,
                                        self.translate)

        return TF.affine(x, 0, (translate_x, translate_y), 1, 0),\
            TF.affine(mask, 0, (translate_x, translate_y), 1, 0)
