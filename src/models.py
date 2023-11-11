from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from config import pretrained_backbone, NUM_CLASSES, device


def prepare_deeplabv3_resnet50(num_classes=NUM_CLASSES, device = device, pretrained_backbone = pretrained_backbone):
    """
    Prepare a DeepLabV3 model for semantic segmentation.

    Parameters
    ----------
    num_classes : int, optional
        Number of classes for semantic segmentation. Defaults to 'NUM_CLASSES' from the config.
    device : torch.device, optional
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
    model = model.to(device)
    return model