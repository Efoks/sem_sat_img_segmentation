import os
import torch

# Define paths to data directories
ORIGINAL_DATA_DIR = 'D:/ann_data'
DATA_DIR = "C:/Users/edvar/PycharmProjects/miniFrance_project/data"
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "train_images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train_masks")
# VAL_IMAGE_DIR = os.path.join(DATA_DIR, "val_images")
# VAL_MASK_DIR = os.path.join(DATA_DIR, "val_masks")

# Define constants
NUM_CLASSES = 16

CLASS_LABELS = {
    0: "No information",
    1: "Urban fabric",
    2: "Industrial, commercial, public, military, private and transport units",
    3: "Mine, dump and construction sites",
    4: "Artificial non-agricultural vegetated areas",
    5: "Arable land (annual crops)",
    6: "Permanent crops",
    7: "Pastures",
    8: "Complex and mixed cultivation patterns",
    9: "Orchards at the fringe of urban classes",
    10: "Forests",
    11: "Herbaceous vegetation associations",
    12: "Open spaces with little or no vegetation",
    13: "Wetlands",
    14: "Water",
    15: "Clouds and shadows"
}

# Encodings for region data
REGIONS = ['D029',
           'D014',
           'D059',
           'D072',
           'D056',
           'D022',
           'D044',
           'D006']
# Original dataset image sizes
# and sub-image sizes that the images will be converted to
ORIGINAL_IMAGE_SIZE = 10_000
PATCH_SIZE = 1_000

# Maximum number of images to leave for the final dataset
MAX_IMAGES = 3_500


# Define paths for model-related data
MODEL_DATA_DIR = "C:/Users/edvar/PycharmProjects/miniFrance_project"
CHECKPOINT_DIR = os.path.join(MODEL_DATA_DIR, "checkpoints")
LOG_DIR = os.path.join(MODEL_DATA_DIR, "logs")

# Determine the device for PyTorch (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model input image size and model general parameters
image_size = (1000, 1000)
backbone = 'resnet50'
pretrained_backbone = True

# Create directories for checkpoints and logs if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def print_config():
    """
    Print the configuration settings for the project.
    """

    print("Configuration:")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Image Size: {image_size}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f'Backbone: {backbone}, is pretrained? {pretrained_backbone}')
    print(f'Device for torch: {device}')