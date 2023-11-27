import os
import torch

# Define paths to data directories
ORIGINAL_DATA_DIR = 'D:\\ann_data'
DATA_DIR = "C:\\Users\\edvar\\PycharmProjects\\miniFrance_project\\data"
SUPERVISED_TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "supervised_dataset", "supervised_images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "supervised_dataset", "masks")
UNSUPERVISED_TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "unsupervised_dataset")

DATA_DIR_V2 = "C:\\Users\\edvar\\PycharmProjects\\miniFrance_project\\data_v2"
SUPERVISED_TRAIN_IMAGE_DIR_V2 = os.path.join(DATA_DIR_V2, "supervised_dataset", "supervised_images")
TRAIN_MASK_DIR_V2 = os.path.join(DATA_DIR_V2, "supervised_dataset", "masks")
UNSUPERVISED_TRAIN_IMAGE_DIR_V2 = os.path.join(DATA_DIR_V2, "unsupervised_dataset")

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
MODEL_DATA_DIR = "C:\\Users\\edvar\\PycharmProjects\\miniFrance_project"

CHECKPOINT_DIR = os.path.join(MODEL_DATA_DIR, "checkpoints")
LOG_DIR = os.path.join(MODEL_DATA_DIR, "logs")

DEEPLABV3_RESNET50_LOG_DIR = os.path.join(LOG_DIR, 'deeplabv3_resnet50_logs')

DEEPLABV3_RESNET101_LOG_DIR = os.path.join(LOG_DIR, 'deeplabv3_resnet101_logs')

DEEPLABV3_MOBILENET_LOG_DIR = os.path.join(LOG_DIR, 'deeplabv3_mobilenet_logs')

UNET_LOG_DIR = os.path.join(LOG_DIR, 'unet_logs')

SAVED_MODELS_DIR = os.path.join(MODEL_DATA_DIR, "saved_models")

DEEPLABV3_RESNET50_SAVE_DIR = os.path.join(SAVED_MODELS_DIR, 'deeplabv3_resnet50_saves')

DEEPLABV3_RESNET101_SAVE_DIR = os.path.join(SAVED_MODELS_DIR, 'deeplabv3_resnet101_saves')

DEEPLABV3_MOBILENET_SAVE_DIR = os.path.join(SAVED_MODELS_DIR, 'deeplabv3_mobilenet_saves')

UNET_SAVE_DIR = os.path.join(SAVED_MODELS_DIR, 'unet_saves')

# Determine the device for PyTorch (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model input image size and model general parameters
resnet50 = {'backbone': 'resnet50',
            'pretrained_backbone': True,
            'num_epochs': 50,
            'batch_size': 2,
            'image_size': (1000, 1000)}

resnet101 = {'backbone': 'resnet101',
            'pretrained_backbone': True,
            'num_epochs': 50,
            'batch_size': 2,
            'image_size': (1000, 1000)}

mobilenet = {'backbone': 'mobilenet_v3_large',
            'pretrained_backbone': True,
            'num_epochs': 50,
            'batch_size': 2,
            'image_size': (1000, 1000)}

unet = {'backbone': 'none',
            'pretrained_backbone': False,
            'num_epochs': 50,
            'batch_size': 2,
            'image_size': (1024, 1024)}

# Create directories for checkpoints and logs if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

os.makedirs(DEEPLABV3_RESNET50_SAVE_DIR, exist_ok=True)

os.makedirs(DEEPLABV3_RESNET101_SAVE_DIR, exist_ok=True)

os.makedirs(DEEPLABV3_MOBILENET_SAVE_DIR, exist_ok=True)

os.makedirs(UNET_SAVE_DIR, exist_ok=True)

os.makedirs(UNET_LOG_DIR, exist_ok=True)

def print_config():
    """
    Print the configuration settings for the project.
    """
    print("Configuration:")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f'Device for torch: {device}')

def print_model_config(model_params):
    for key in model_params:
        print(f'Configuration: {key}, Value: {model_params[key]}')
