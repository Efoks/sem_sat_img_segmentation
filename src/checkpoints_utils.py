import torch
import config

def save_model_checkpoint(model, optimizer, epoch, loss, checkpoint_path = config.CHECKPOINT_DIR):
    """
    Save a model's state, optimizer state, training epoch, and loss to a checkpoint file.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be saved.
    optimizer : torch.optim.Optimizer
        The optimizer used to train the model.
    epoch : int
        The current training epoch at the time of saving.
    loss : float
        The value of the loss at the time of saving.
    checkpoint_path : str, optional
        The file path where the checkpoint will be saved.
        Defaults to the value in the 'config.CHECKPOINT_DIR'.

    Returns
    -------
    None
    """

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_path)

def load_model_checkpoint(model, optimizer, checkpoint_path = config.CHECKPOINT_DIR):
    """
    Load a model's state, optimizer state, training epoch, and loss from a checkpoint file.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be loaded with the saved state.
    optimizer : torch.optim.Optimizer
        The optimizer to be loaded with the saved state.
    checkpoint_path : str, optional
        The file path from which to load the checkpoint.
        Defaults to the value in the 'config.CHECKPOINT_DIR'.

    Returns
    -------
    epoch : int
        The training epoch saved in the checkpoint.
    loss : float
        The loss value saved in the checkpoint.
    """

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return epoch, loss