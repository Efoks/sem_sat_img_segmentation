import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import rasterio
from rasterio.plot import  reshape_as_image
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import logging
import src.config as config
import src.utils as utils

# Mean and standard deviation for image normalization, that are used for pretrained DeepLabV3
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MiniFranceDataset(Dataset):
    """
       Custom dataset class for handling the MiniFrance dataset, which includes supervised and unsupervised images.

       Attributes:
           sup_image_dir (str): Directory containing supervised training images.
           mask_dir (str): Directory containing masks for supervised images.
           unsup_image_dir (str): Directory containing unsupervised training images.
           transformation (callable, optional): A function/transform that takes in an image and returns a transformed version.

       Methods:
           __len__: Returns the total number of images in the dataset.
           __getitem__: Retrieves an image and its corresponding mask by index.
    """
    def __init__(self, supervised_train_images_dir=config.SUPERVISED_TRAIN_IMAGE_DIR,
                        train_masks_dir=config.TRAIN_MASK_DIR,
                        unsupervised_train_images_dir=config.UNSUPERVISED_TRAIN_IMAGE_DIR,
                        transform=None):

        self.sup_image_dir = supervised_train_images_dir
        self.mask_dir = train_masks_dir
        self.unsup_image_dir = unsupervised_train_images_dir

        self.image_files = os.listdir(supervised_train_images_dir) + os.listdir(unsupervised_train_images_dir)

        self.transformation = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Retrieves an image by index, along with its mask if it's a supervised image

        image_filename = self.image_files[idx]

        if os.path.exists(os.path.join(self.sup_image_dir, image_filename)):

            image_path = os.path.join(self.sup_image_dir, image_filename)
            mask_path = os.path.join(self.mask_dir,
                                         f"mask_{os.path.splitext(image_filename)[0]}.tif")

            image = reshape_as_image(rasterio.open(image_path).read())

            if self.transformation:
                image = self.transformation(image)

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
            image_path = os.path.join(self.unsup_image_dir, image_filename)

            image = reshape_as_image(rasterio.open(image_path).read())

            if self.transformation:
                image = self.transformation(image)

            mask = [] # For compatability with DataLoader (does not accept None)
        return image, mask

def train_val_stratified_split(idx, mask_path = config.TRAIN_MASK_DIR, show_cluster = False):
    class_frequencies = utils.calculate_class_frequencies(mask_path)
    clusters = utils.cluster_images(class_frequencies)

    if show_cluster:
        utils.plot_cluster(class_frequencies, clusters)

    X_train_idx, X_val_idx, _, _ = train_test_split(idx, idx, stratify=clusters)
    return X_train_idx, X_val_idx

def train_val_split(idx):
    X_train_idx, X_val_idx, _, _ = train_test_split(idx, idx)
    return X_train_idx, X_val_idx

def create_data_loaders(batch_size, perform_stratiication = False, show_cluster = False, data_2 = False, model = 'deeplab'):
    """
    Creates data loaders for the MiniFrance dataset for pre-trained DeepLab Models.

    Args:
        batch_size (int): The size of each batch during training.
        perform_stratiication (bool): Whether to stratify the dataset.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, DataLoader]: Returns four data loaders -
        two each for supervised and unsupervised data (training and validation).
    """
    if model == 'deeplab':
        # Data normalization as defined in DeepLabV3 documentation
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif model == 'unet':
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((1024, 1024))
        ])
    else:
        print('No model was found')

    if data_2:
        dataset = MiniFranceDataset(supervised_train_images_dir=config.SUPERVISED_TRAIN_IMAGE_DIR_V2,
                                    train_masks_dir=config.TRAIN_MASK_DIR_V2,
                                    unsupervised_train_images_dir=config.UNSUPERVISED_TRAIN_IMAGE_DIR_V2,
                                    transform=data_transforms)

        supervised_idx = [i for i in range(len(os.listdir(config.SUPERVISED_TRAIN_IMAGE_DIR_V2)))]
        unsupervised_idx = [i + len(supervised_idx) for i in
                            range(len(os.listdir(config.UNSUPERVISED_TRAIN_IMAGE_DIR_V2)))]
    else:
        dataset = MiniFranceDataset(transform=data_transforms)

        supervised_idx = [i for i in range(len(os.listdir(config.SUPERVISED_TRAIN_IMAGE_DIR)))]
        unsupervised_idx = [i + len(supervised_idx) for i in
                            range(len(os.listdir(config.UNSUPERVISED_TRAIN_IMAGE_DIR)))]

    size = len(dataset)

    if perform_stratiication:
        train_supervised_idx, val_supervised_idx = train_val_stratified_split(supervised_idx, show_cluster = show_cluster)
        print('Data clusterized and stratified')
    else:
        train_supervised_idx, val_supervised_idx = train_val_split(supervised_idx)

    train_unsupervised_idx, val_unsupervised_idx = train_val_split(unsupervised_idx)

    supervised_sampler_train = SubsetRandomSampler(train_supervised_idx)
    unsupervised_sampler_train = SubsetRandomSampler(train_unsupervised_idx)

    supervised_sampler_val = SubsetRandomSampler(val_supervised_idx)
    unsupervised_sampler_val = SubsetRandomSampler(val_unsupervised_idx)

    sup_loader_train = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=supervised_sampler_train,
                                drop_last=True)

    sup_loader_val = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=supervised_sampler_val,
                                drop_last=True)

    unsup_loader_train = DataLoader(dataset,
                                  batch_size=batch_size,
                                  sampler=unsupervised_sampler_train,
                                  drop_last=True)

    unsup_loader_val = DataLoader(dataset,
                                  batch_size=batch_size,
                                  sampler=unsupervised_sampler_val,
                                  drop_last=True)


    return sup_loader_train, sup_loader_val, unsup_loader_train, unsup_loader_val




if __name__ == "__main__":
    batch_size = 2
    unsupervised_ratio = 0.5

    supervised_loader, _, unsupervised_loader, _ = create_data_loaders(batch_size,
                                                                               perform_stratiication=True,
                                                                               show_cluster=True)

    utils.plot_images_and_masks(supervised_loader, unsupervised_loader)
    # utils.plot_class_distribution(supervised_loader)


