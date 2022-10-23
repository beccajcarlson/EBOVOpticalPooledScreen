import copy
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from transforms import TRANSFORMS
from config_tools import get_device
from model import ConvAutoencoderWithHead


def SupervisedCellsDataset(train_labeled_set,
                           test_labeled_set,
                           labeled_samples_npy, labeled_masks_npy,
                           channel_first=False,
                           n_samples_per_group=5,
                           denoising=False,
                           n_transforms=4):
    """Dataset for supervised cell image extraction from processed DataFrames

    Args:
        train_labeled_set (pandas DataFrame): Dataframe of train labeled samples,
            with index_loc column referring to the index in labeled_samples_npy
            containing the cell sample
        test_labeled_set (pandas DataFrame): Dataframe of test labeled samples,
            with index_loc column referring to the index in labeled_samples_npy
            containing the cell sample
        labeled_samples_npy (numpy array): Array of cell samples in samples x channels format
        labeled_masks_npy (numpy array): Array of masks in samples x channels format
        channel_first (bool, optional): Channel first in images. Defaults to False.
        n_samples_per_group (int, optional): Number of samples per training batch. Defaults to 5.
        denoising (bool, optional): Whether to return different inputs and targets
            for autoencoder transformations. Defaults to False.
        n_transforms (int, optional): Number of augmented
            images to generate per cell. Defaults to 4.

    Returns:
        SupervisedCellsPseudoDataset: A dataset of supervised (hand-labeled) cells
    """
    return SupervisedCellsPseudoDataset(train_labeled_set,
                                        test_labeled_set,
                                        labeled_samples_npy,
                                        labeled_masks_npy,
                                        n_samples_per_group,
                                        channel_first, denoising,
                                        n_transforms)


class SupervisedCellsPseudoDataset:
    """Pseudo-dataset for supervised cell generation

    Infinitely many cells are generated from a small initial
    batch via transforms and augmentations
    """

    def __init__(self, train_labeled_set,
                 test_labeled_set,
                 labeled_samples_npy,
                 labeled_masks_npy,
                 n_samples_per_group,
                 channel_first, denoising,
                 n_transforms):
        """See SupervisedCellsDataset function definition
        for argument descriptions
        """
        self.denoising = denoising
        self.channel_first = channel_first
        self.raw_ims = np.load(labeled_samples_npy)
        self.raw_masks = np.load(labeled_masks_npy)
        self.train_labeled_set = train_labeled_set
        self.n_transforms = n_transforms
        self.n_samples_per_group = n_samples_per_group
        self.stratified_train_set = {}

        # Generate stratified training sets by splitting on label
        for label, g in self.train_labeled_set.groupby(["label"]):
            self.stratified_train_set[int(label)] = g.reset_index(drop=True)

        self.test_labeled_set = test_labeled_set

    def get_train_samples(self):
        """Gets a set of n_transforms x n_samples_per_group training samples

        Returns:
            torch Tensor, torch Tensor, torch Tensor, torch Tensor:
                images, masks, targets, labels
        """
        # Temporary lists for storing arrays
        img_array, mask_array, target_array, labels = [], [], [], []

        # Iterate over cells of each label (phenotype)
        for label, df in self.stratified_train_set.items():
            # Select a random subset of each label, in the amount of n_samples_per_group
            selected_subsamples = df.iloc[np.random.choice(np.arange(len(df)),
                                                           self.n_samples_per_group,
                                                           replace=False)]

            # Extract images, masks and apply transforms
            img_set, mask_set = self.extract_ims_masks(
                selected_subsamples, labels=False)
            imgs, masks, targets = self.apply_transforms_multi(
                img_set, mask_set)

            img_array.append(imgs)
            mask_array.append(masks)
            target_array.append(targets)
            # Labels are the same for all members of a transform group
            # that is - all augmented images of a source image share a phenotype
            labels.extend(
                [label] * (self.n_samples_per_group * self.n_transforms))

        # Stack image, mask, and target arrays
        img_array, mask_array, target_array = \
            torch.vstack(img_array), torch.vstack(
                mask_array), torch.vstack(target_array)

        if not self.channel_first:
            img_array = torch.moveaxis(img_array, -1, 1)
            mask_array = torch.moveaxis(mask_array, -1, 1)
            target_array = torch.moveaxis(target_array, -1, 1)

        img_data = img_array
        mask_data = mask_array
        target_data = target_array
        label_data = torch.Tensor(labels)

        return img_data, mask_data, target_data, label_data

    def extract_ims_masks(self, df, labels=False):
        """Extract images and masks given a subset DataFrame

        Args:
            df (pandas DataFrame): Subset of cells to select images and masks from
            labels (bool, optional): Whether to return labels. Defaults to False.

        Returns:
            torch Tensor, torch Tensor, (torch Tensor): images, masks, (labels)
        """
        # Extract indices in raw data arrays of the images and masks
        indices = df["index_loc"]
        ims, masks = torch.Tensor(self.raw_ims[indices]), torch.Tensor(
            self.raw_masks[indices])

        if labels:
            return ims, masks, torch.Tensor(df["label"].to_numpy())
        else:
            return ims, masks

    def apply_transforms_multi(self, imgs, masks, staple=True):
        """Apply transformations to a set of images and masks

        Args:
            imgs (array): _description_
            masks (array): _description_
            staple (bool, optional): Whether to apply the transforms
                simultaneously to both the image and mask
                or separately to each. Defaults to True.

        Returns:
            torch Tensor, torch Tensor, torch Tensor: images, masks, and targets
        """
        # For each unique image and mask, apply the desired transform sequence
        img_array = []
        mask_array = []
        target_array = []

        # Iterate over the set of images, perform the transforms randomly
        for i in range(len(imgs)):

            img = imgs[i, ...]
            mask = masks[i, ...]

            # For each of the desired augmented images, apply random transforms
            for _ in range(self.n_transforms):
                # Randomly select a subset of transforms to apply
                cutoff = np.random.choice(len(TRANSFORMS) + 1, 1)
                subset_transforms = np.random.choice(
                    len(TRANSFORMS), cutoff, replace=False)
                t_img, t_mask = img.clone(), mask.clone()

                # Sequentially apply the random set of transforms
                for tf_idx in subset_transforms:
                    if not staple:
                        t_img, t_mask = TRANSFORMS[tf_idx](t_img, t_mask)
                    else:
                        combo = TRANSFORMS[tf_idx](
                            torch.concat([t_img, t_mask], dim=0))
                        t_img, t_mask = combo[:6, ...], torch.where(
                            combo[6:, ...] > 0, 1, 0)

                img_array.append(t_img)
                mask_array.append(t_mask)

                # If denoising, target is the original image, not the augmented one
                if self.denoising:
                    target_array.append(t_img)
                else:
                    target_array.append(img.clone())

        return torch.stack(img_array), torch.stack(mask_array), torch.stack(target_array)

    def get_dataset(self, test=False):
        """Returns the images and masks for the train or test raw dataset

        Args:
            test (bool, optional): Whether the desired dataset is Test. Defaults to False.

        Returns:
            torch Tensor, torch Tensor, torch Tensor: images, masks, labels
        """
        if test:
            labeled_group = self.test_labeled_set
        else:
            labeled_group = self.train_labeled_set

        return self.extract_ims_masks(labeled_group, labels=True)


def evaluate_model(model, dataset, device, test=True):
    """Evaluates a model on the test set

    Args:
        model (model.ConvAutoencoderWithHead): The autoencoder to evaluate
        dataset (SupervisedCellsPseudoDataset): A dataset of hand-labeled cells
        device (str): Device on which to perform inference
        test (bool, optional): Whether to use test images. Defaults to True.

    Returns:
        float, float, List[int], List[int]: Overall accuracy, balanced accuracy,
            correct labels, predicted labels
    """
    corrects = []
    pred = []

    model.eval()
    with torch.no_grad():
        # Get dataset, predict labels from model
        images, _, true_labels = dataset.get_dataset(test)
        images = images.squeeze(0).to(device)
        _, labels, _ = model(images)

        labels = (labels.argmax(axis=1) + 1).detach().cpu().numpy().ravel()

        pred = labels.tolist()
        corrects = true_labels.int().tolist()
    model.train()

    return float(accuracy_score(corrects, pred)),\
        float(balanced_accuracy_score(corrects, pred)),\
        corrects, pred


def load_supervised_from_unsupervised(pretrained_model_path, hidden_units):
    """Loads weights from pretrained unsupervised model into supervised model

    Args:
        pretrained_model_path (str): Path to pretrained ConvAutoencoder state
        hidden_units (int): Number of hidden units in classification head

    Returns:
        ConvAutoencoderWithHead: Semi-supervised model with weights preloaded
    """
    device = get_device()
    model = ConvAutoencoderWithHead(hidden=hidden_units)
    model.apply(ConvAutoencoderWithHead.init_weights)
    model.load_state_dict(torch.load(pretrained_model_path,
                                     map_location=device),
                          strict=False)
    model.to(device)
    return model


def train_supervised_model(supervised_ds, model, rng, n_epochs=100, fine_tuning=100):
    """Trains the supervised portion of a semi-supervised autoencoder

    Args:
        supervised_ds (SupervisedCellsPseudoDataset): A dataset of hand-labeled cells
        model (model.ConvAutoencoderWithHead): The autoencoder to train
        rng (np.random.Generator): Random number generator
        n_epochs (int, optional): Number of epochs to train for. Defaults to 100.
        fine_tuning (int, optional): Number of iterations of fine-tuning to perform.
            Defaults to 100.

    Returns:
        List[float], List[(float, float, float)],
        ConvAutoencoderWithHead, (ConvAutoencoderWithHead, int),
        (ConvAutoencoderWithHead, int), (ConvAutoencoderWithHead, int):
            Model losses per epoch; model accuracy, balanced accuracy, and
            average accuracy per epoch on test set; final model; model with
            best test accuracy and epoch attained; model with best balanced
            test accuracy and epoch attained; model with best average
            balanced/overall test accuracy and epoch attained
    """

    # Get device to use
    device = get_device()
    model = model.to(device)

    # Loss function
    criterion2 = torch.nn.NLLLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Overall Stats
    losses = []
    accuracies = []
    best_balanced_model = None
    best_average_model = None
    best_acc_model = None

    pbar = tqdm(range(1, n_epochs + 1))
    for epoch in pbar:
        # Monitor training loss
        train_loss2 = 0.0

        # Fine-Tuning Model using labeled dataset
        for _ in tqdm(range(fine_tuning)):
            s_images, _, _, s_labels = supervised_ds.get_train_samples()

            # Perform Batch Shuffle
            tot_batch_size = s_images.shape[0]
            indices = np.arange(tot_batch_size)
            rng.shuffle(indices)

            data = s_images[indices]
            labels = s_labels[indices]

            labels = labels.float().to(device)
            labeled_indices = torch.nonzero(~labels.isnan()).squeeze(0).to(device)
            only_labels = labels[labeled_indices].squeeze()

            images = data.to(device)

            optimizer.zero_grad()
            _, pred_labels, _ = model(images)
            loss2 = criterion2(pred_labels[labeled_indices].squeeze(),
                               only_labels.long() - 1)
            loss2.backward()
            optimizer.step()
            train_loss2 += loss2.item()

        model.eval()

        acc, bal_acc, _, _ = evaluate_model(model, supervised_ds,
                                            device, test=True)

        avg_acc = (acc + bal_acc) / 2

        # Populate list of best models, along with accompanying epoch
        if (best_acc_model is None) or\
           (bal_acc >= max(accuracies, key=lambda x: x[0])[0]):
            best_acc_model = (copy.deepcopy(model), epoch)

        if (best_balanced_model is None) or\
           (bal_acc >= max(accuracies, key=lambda x: x[1])[1]):
            best_balanced_model = (copy.deepcopy(model), epoch)

        if (best_average_model is None) or\
           (avg_acc >= max(accuracies, key=lambda x: x[2])[2]):
            best_average_model = (copy.deepcopy(model), epoch)

        accuracies.append((acc, bal_acc, avg_acc))

        model.train()

        train_loss2 = train_loss2 / fine_tuning
        losses.append(train_loss2)

        pbar.set_description('Epoch: {} \tTraining Loss 1: {:.6f}\tTest Accs: {:.4f},{:.4f}'
                             .format(epoch, train_loss2, acc, bal_acc))

    return losses, accuracies, model,\
        best_acc_model, best_balanced_model, best_average_model
