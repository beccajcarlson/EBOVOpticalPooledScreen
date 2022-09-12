import torch
import numpy as np

from transforms import TRANSFORMS


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
