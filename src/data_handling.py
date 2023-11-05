import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import config
import rasterio
from rasterio.plot import  reshape_as_image
import torch.nn.functional as F
import logging
import utils

# Mean and standard deviation for image normalization, that are used for pretrained DeepLabV3
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MiniFranceDataset(Dataset):
    def __init__(self, train_images_dir=config.TRAIN_IMAGE_DIR, train_masks=config.TRAIN_MASK_DIR, transform=None):
        """
        Custom PyTorch dataset for MiniFrance images and masks.

        Parameters
        ----------
        train_images_dir : str, optional
            Directory containing input images. Defaults to config.TRAIN_IMAGE_DIR.
        train_masks : str, optional
            Directory containing corresponding masks. Defaults to config.TRAIN_MASK_DIR.
        transform : callable, optional
            A transform to apply to the images.

        Attributes
        ----------
        image_dir : str
            Directory containing input images.
        mask_dir : str
            Directory containing corresponding masks.
        image_files : list
            List of image file names.
        transformation : callable
            Transformation to apply to the images.
        """

        self.image_dir = train_images_dir
        self.mask_dir = train_masks
        self.image_files = os.listdir(train_images_dir)
        self.transformation = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir,
                                  image_filename)
        mask_path = os.path.join(self.mask_dir,
                                 f"mask_{os.path.splitext(image_filename)[0]}.tif")

        image = reshape_as_image(rasterio.open(image_path).read())

        if self.transformation:
            image = self.transformation(image)

        if os.path.exists(mask_path):
            mask = rasterio.open(mask_path).read(1)
            mask = torch.from_numpy(mask).long()  # Convert numpy array to LongTensor

            # Check if the mask contains valid class indices
            if not torch.all((mask >= 0) & (mask < config.NUM_CLASSES)):
                raise ValueError(f"Mask values must be between 0 and {config.NUM_CLASSES - 1}")

            # One hot encode to convert masks shape into [num_classes, size_x, size_y]
            # from [num_channels, size_x, size_y]
            mask = F.one_hot(mask, config.NUM_CLASSES)
            mask = mask.permute(2, 0, 1).float()
        else:
            # Used for unsupervised data part
            # Create a zero mask with shape [num_classes, H, W]
            mask = torch.zeros((config.NUM_CLASSES, image.shape[1], image.shape[2]), dtype=torch.float32)

        return image, mask


def create_data_loaders(batch_size, unsupervised_ratio):
    """
    Create supervised and unsupervised data loaders for training.

    Parameters
    ----------
    batch_size : int
        Batch size for both supervised and unsupervised data loaders.
    unsupervised_ratio : float
        Ratio of unsupervised samples to total samples.

    Returns
    -------
    sup_loader : DataLoader
        Supervised data loader.
    unsup_loader : DataLoader
        Unsupervised data loader.
    """

    # Data normalization as defined in DeepLabV3 documentation
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataset = MiniFranceDataset(transform=data_transforms)
    size = len(dataset)

    supervised_idx = [idx for idx in range(size) if os.path.exists(
        os.path.join(dataset.mask_dir, f'mask_{os.path.splitext(dataset.image_files[idx])[0]}.tif'))]

    # Used for checking which images have mask (used for supervised learning) vs
    # images which do not have masks and will be used for unsupervised learning
    unsupervised_idx = [idx for idx in range(size) if idx not in supervised_idx]
    num_unsupervised_samples = int(unsupervised_ratio * len(unsupervised_idx))

    supervised_sampler = SubsetRandomSampler(supervised_idx)
    unsupervised_sampler = SubsetRandomSampler(unsupervised_idx[:num_unsupervised_samples])

    sup_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=supervised_sampler)

    unsup_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              sampler=unsupervised_sampler)

    return sup_loader, unsup_loader


if __name__ == "__main__":
    batch_size = 2
    unsupervised_ratio = 0.5

    supervised_loader, unsupervised_loader = create_data_loaders(batch_size, unsupervised_ratio)

    utils.plot_images_and_masks(supervised_loader, unsupervised_loader)
    # utils.plot_class_distribution(supervised_loader)


