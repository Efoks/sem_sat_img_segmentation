import logging
import os
from datetime import datetime
from config import LOG_DIR
import hyperparameter_config as hc

def setup_logging(log_dir = LOG_DIR):
    """
    Set up logging for training and save logs to a file.

    Parameters
    ----------
    log_dir : str, optional
        Directory where log files will be saved. Defaults to 'LOG_DIR' from the config.

    Returns
    -------
    logger : logging.Logger
        The configured logger.
    """

    log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(filename=log_file,
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger()

    return logger

def log_hyperparameters(logger):
    """
    Log hyperparameters settings to the logger at the start of the modelling.

    Parameters
    ----------
    logger : logging.Logger
        The logger to which hyperparameters will be logged.

    Returns
    -------
    None
    """

    logger.info("Hyperparameters Settings:")
    logger.info(f'Batch Size: {hc.batch_size}; Number of Epochs: {hc.num_epochs}; Learning Rate: {hc.learning_rate}; Momentum: {hc.momentum}; Dampening: {hc.dampening}')

def log_training_progress(logger, epoch, total_epochs, train_loss, val_loss, train_metrics, val_metrics):
    """
    Log training progress including loss and metrics for each epoch.

    Parameters
    ----------
    logger : logging.Logger
        The logger used to log training progress.
    epoch : int
        Current epoch number.
    total_epochs : int
        Total number of training epochs.
    train_loss : float
        Training loss for the current epoch.
    val_loss : float
        Validation loss for the current epoch.
    train_metrics : dict
        Training metrics for the current epoch.
    val_metrics : dict
        Validation metrics for the current epoch.

    Returns
    -------
    None
    """

    logger.info(f"Epoch [{epoch}/{total_epochs}]:")
    logger.info(f"Train Loss: {train_loss:.4f}")
    logger.info(f"Validation Loss: {val_loss:.4f}")
    logger.info(f"Train Metrics: {train_metrics}")
    logger.info(f"Validation Metrics: {val_metrics}")

def log_batch_progress(logger, epoch, total_epochs, batch_idx, total_batches, train_loss):
    """
    Log progress for each batch during training.

    Parameters
    ----------
    logger : logging.Logger
        The logger used to log batch progress.
    epoch : int
        Current epoch number.
    total_epochs : int
        Total number of training epochs.
    batch_idx : int
        Current batch index.
    total_batches : int
        Total number of batches in the current epoch.
    train_loss : float
        Training loss for the current batch.

    Returns
    -------
    None
    """

    logger.info(f"Epoch [{epoch}/{total_epochs}], Batch [{batch_idx}/{total_batches}]:")
    logger.info(f"Train Loss: {train_loss:.4f}")
