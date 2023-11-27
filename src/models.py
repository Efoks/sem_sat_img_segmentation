from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from src.config import NUM_CLASSES, device
import src.config as config
from src.u_net import UNet


def prepare_deeplabv3_resnet50(num_classes=NUM_CLASSES, device_to_run = device,
                               pretrained_backbone = config.resnet50['pretrained_backbone']):
    """
    Prepare a DeepLabV3 model for semantic segmentation.

    Parameters
    ----------
    num_classes : int, optional
        Number of classes for semantic segmentation. Defaults to 'NUM_CLASSES' from the config.
    device_to_run : torch.device, optional
        Device (CPU or GPU) on which to place the model. Defaults to 'device' from the config.
    pretrained_backbone : bool, optional
        Whether to use a pretrained ResNet-50 backbone. Defaults to 'pretrained_backbone' from the config.

    Returns
    -------
    model : torch.nn.Module
        The configured DeepLabV3 model.
    """

    model = deeplabv3_resnet50(pretrained = pretrained_backbone)
    model.classifier = DeepLabHead(2048, num_classes)
    model = model.to(device_to_run)
    return model

def prepare_deeplabv3_resnet101(num_classes=NUM_CLASSES, device_to_run = device,
                               pretrained_backbone = config.resnet101['pretrained_backbone']):
    """
    Prepare a DeepLabV3 model for semantic segmentation.

    Parameters
    ----------
    num_classes : int, optional
        Number of classes for semantic segmentation. Defaults to 'NUM_CLASSES' from the config.
    device_to_run : torch.device, optional
        Device (CPU or GPU) on which to place the model. Defaults to 'device' from the config.
    pretrained_backbone : bool, optional
        Whether to use a pretrained ResNet-101 backbone. Defaults to 'pretrained_backbone' from the config.

    Returns
    -------
    model : torch.nn.Module
        The configured DeepLabV3 model.
    """

    model = deeplabv3_resnet101(pretrained = pretrained_backbone)
    model.classifier = DeepLabHead(2048, num_classes)
    model = model.to(device_to_run)
    return model

def prepare_deeplabv3_mobilenet(num_classes=NUM_CLASSES, device_to_run = device,
                               pretrained_backbone = config.mobilenet['pretrained_backbone']):
    """
    Prepare a DeepLabV3 model for semantic segmentation.

    Parameters
    ----------
    num_classes : int, optional
        Number of classes for semantic segmentation. Defaults to 'NUM_CLASSES' from the config.
    device_to_run : torch.device, optional
        Device (CPU or GPU) on which to place the model. Defaults to 'device' from the config.
    pretrained_backbone : bool, optional
        Whether to use a pretrained MobileNet backbone. Defaults to 'pretrained_backbone' from the config.

    Returns
    -------
    model : torch.nn.Module
        The configured DeepLabV3 model.
    """

    model = deeplabv3_mobilenet_v3_large(pretrained = pretrained_backbone)
    model.classifier = DeepLabHead(960, num_classes)
    model = model.to(device_to_run)
    return model

def prepare_u_net(num_classes=NUM_CLASSES, device_to_run = device):

    model = UNet(channels_in = 3, channels_out = num_classes)
    model = model.to(device_to_run)
    return model