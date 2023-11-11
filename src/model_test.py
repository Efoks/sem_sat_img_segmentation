import torch
import torch.nn as nn
import torch.optim as optim
from models import prepare_model
from checkpoints_utils import save_model_checkpoint, load_model_checkpoint
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from logging_functions import setup_logging, log_hyperparameters, log_batch_progress, log_training_progress
import torch.nn.functional as F
import config
import utils

class MockDataset(Dataset):
    def __init__(self, size, num_classes, image_size, batch_size, transform=None):
        """
        Initialize a mock dataset for testing.

        Parameters
        ----------
        size : int
            Number of samples in the dataset.
        num_classes : int
            Number of classes in the dataset.
        image_size : tuple
            Image size (height, width).
        batch_size : int
            Batch size for DataLoader.
        transform : callable, optional
            Transformation to apply to the data. Defaults to None.
        """

        self.data = torch.rand(size, 3, *image_size).to('cuda')
        self.labels = torch.randint(0, num_classes, (size, 1, *image_size)).to('cuda')
        self.batch_size = batch_size
        self.trasformation = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        mask = F.one_hot(self.labels[idx].squeeze(1), config.NUM_CLASSES)
        mask = mask.permute(0, 3, 1, 2).float().squeeze(0)
        return self.trasformation(image), mask


def create_mock_data_loader(inputs, targets, batch_size):
    """
    Create a data loader for a mock dataset.

    Parameters
    ----------
    inputs : torch.Tensor
        Input data.
    targets : torch.Tensor
        Target data.
    batch_size : int
        Batch size for DataLoader.

    Returns
    -------
    data_loader : torch.utils.data.DataLoader
        DataLoader for the mock dataset.
    """

    dataset = TensorDataset(inputs,
                            targets)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True)
    return data_loader

def test_evaluate_model(model, data_loader):
    """
    Test and evaluate the model using a mock data loader and display the IoU score.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be tested and evaluated.
    data_loader : torch.utils.data.DataLoader
        DataLoader for the mock dataset.

    Returns
    -------
    None
    """

    iou_score = utils.evaluate_model(model, data_loader)
    print(f"Mean IoU Score: {iou_score:.4f}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    num_samples = 20
    num_classes = 16
    image_size = (1000, 1000)
    batch_size = 2

    model = prepare_model(num_classes=num_classes,
                          device='cuda')

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = transforms.Compose([
        transforms.Normalize(mean, std)
    ])

    mock_dataset = MockDataset(size=num_samples,
                               num_classes=num_classes,
                               image_size=image_size,
                               batch_size=batch_size,
                               transform= data_transforms)

    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_parameters)

    criterion = nn.BCEWithLogitsLoss()


    num_epochs = 3

    logger = setup_logging()
    log_hyperparameters(logger)

    for epoch in range(num_epochs):

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            data_loader = DataLoader(mock_dataset,
                                     batch_size=batch_size,
                                     shuffle=True)

            for batch_idx, (data, target) in enumerate(data_loader):
                optimizer.zero_grad()
                output = model(data.to('cuda'))
                output_predictions = output['out']

                loss = criterion(output_predictions.float(), target.to('cuda').float())
                log_batch_progress(logger, epoch + 1, num_epochs, batch_idx + 1, num_samples / batch_size, loss)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

        test_evaluate_model(model, data_loader)