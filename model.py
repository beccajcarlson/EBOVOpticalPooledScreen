import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder for generating cell images
    """

    def __init__(self):
        """Initialize Convolutional Autoencoder
        """
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(6, 32, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=2, padding=1)

        # Decoder
        self.upconv1 = nn.Conv2d(512, 256, 3, padding=1)
        self.upconv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.upconv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.upconv4 = nn.Conv2d(64, 32, 3, padding=1)
        self.upconv5 = nn.Conv2d(32, 6, 5, padding=2)

        self.t_up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Encode image
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))

        # Obtain Latent Space Representation
        x_flat = torch.flatten(x, start_dim=1)

        # Decode image
        x = F.leaky_relu(self.upconv1(x))
        x = self.t_up(x)
        x = F.leaky_relu(self.upconv2(x))
        x = self.t_up(x)
        x = F.leaky_relu(self.upconv3(x))
        x = self.t_up(x)
        x = F.leaky_relu(self.upconv4(x))
        x = self.t_up(x)
        x = F.leaky_relu(self.upconv5(x))
        x = self.t_up(x)

        return x, x_flat

    @staticmethod
    def init_weights(m, bias=0.01):
        """Initialize weights of autoencoder

        Args:
            m (torch nn Layer): Neural network layer
            bias (float, optional): Constant to set biases. Defaults to 0.01.
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(
                m.weight, mode='fan_in', nonlinearity='leaky_relu')
            m.bias.data.fill_(bias)


class ConvAutoencoderWithHead(nn.Module):
    """Convolutional Autoencoder for generating cell images
    Also used for predicting cell phenotype
    """

    def __init__(self, hidden=256):
        """Initialize Convolutional Autoencoder

        Args:
            hidden (int, optional): Number of hidden units in classification head. Defaults to 256.
        """
        super(ConvAutoencoderWithHead, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(6, 32, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=2, padding=1)

        # Classifier
        self.lin1 = nn.Linear(2048, hidden)
        self.lin2 = nn.Linear(hidden, 4)
        self.softmax = nn.LogSoftmax(dim=1)

        # Decoder
        self.upconv1 = nn.Conv2d(512, 256, 3, padding=1)
        self.upconv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.upconv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.upconv4 = nn.Conv2d(64, 32, 3, padding=1)
        self.upconv5 = nn.Conv2d(32, 6, 5, padding=2)

        self.t_up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Encode image
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))

        # Obtain latent space representation and apply classifier
        x_flat = torch.flatten(x, start_dim=1)
        y = self.lin1(x_flat)
        y = self.lin2(y)
        y = self.softmax(y)

        # Decode image
        x = F.leaky_relu(self.upconv1(x))
        x = self.t_up(x)
        x = F.leaky_relu(self.upconv2(x))
        x = self.t_up(x)
        x = F.leaky_relu(self.upconv3(x))
        x = self.t_up(x)
        x = F.leaky_relu(self.upconv4(x))
        x = self.t_up(x)
        x = F.leaky_relu(self.upconv5(x))
        x = self.t_up(x)

        return x, y, x_flat

    def forward_only_classifier(self, x_flat):
        """Run classification head on a latent space embedding
        """
        y = self.lin1(x_flat)
        y = self.lin2(y)
        y = self.softmax(y)
        return y

    def forward_only_decoder(self, x_flat):
        """Run reconstruction decoder on latent space embedding
        """
        x = F.leaky_relu(self.upconv1(x_flat))
        x = self.t_up(x)
        x = F.leaky_relu(self.upconv2(x))
        x = self.t_up(x)
        x = F.leaky_relu(self.upconv3(x))
        x = self.t_up(x)
        x = F.leaky_relu(self.upconv4(x))
        x = self.t_up(x)
        x = F.leaky_relu(self.upconv5(x))
        x = self.t_up(x)
        return x


def my_custom_mse(output, target, mask):
    """Custom MSE applying loss only over masked area

    Args:
        output (torch Tensor): Model output reconstruction
        target (torch Tensor): True image
        mask (torch Tensor): True mask

    Returns:
        torch Tensor: MSE loss over mask area
    """
    return torch.square(output-target).sum() / ((mask == 1).sum())


def weighted_mse_loss(output, target, weight):
    """Weighted MSE applying weights per-image

    Args:
        output (torch Tensor): Model output reconstruction
        target (torch Tensor): True image
        weight (torch Tensor): Weights to apply to each target

    Returns:
        torch Tensor: Weighted MSE loss
    """
    return (weight * (output - target) ** 2).mean()


def prepare_classification_head(pretrained_model="my_model.pth", device='cpu', hidden=256):
    """Prepares an autoencoder with classification head from a pretrained one without

    Args:
        pretrained_model (str or ConvAutoencoder, optional): Pretrained autoencoder. Defaults to my_model.pth.
        device (str, optional): Device to store model. Defaults to 'cpu'.
        hidden (int, optional): Number of hidden dimensions in classification head. Defaults to 256.

    Returns:
        ConvAutoencoderWithHead: Autoencoder having weights of pretrained model
    """
    pretrained_model.to(device)
    with_head = ConvAutoencoderWithHead(hidden=hidden)
    with_head.apply(ConvAutoencoder.init_weights)

    # Load weights
    if isinstance(pretrained_model, str):
        with_head.load_state_dict(torch.load(pretrained_model, map_location=device),
                                  strict=False)
    else:
        with_head.load_state_dict(pretrained_model.state_dict(), strict=False)

    with_head.to(device)

    return with_head
