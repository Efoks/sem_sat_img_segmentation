import torch
import torch.nn as nn
import src.config as config
import matplotlib.pyplot as plt
import os
import rasterio
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Any
from collections import namedtuple
import math

def plot_images_and_masks(supervised_loader: object, unsupervised_loader: object) -> object:
    """
    Plot supervised and unsupervised images and masks for visualization.

    Parameters
    ----------
    supervised_loader : torch.utils.data.DataLoader
        DataLoader for supervised images and masks.
    unsupervised_loader : torch.utils.data.DataLoader
        DataLoader for unsupervised images.

    Returns
    -------
    None
    """

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Figure 3. Images and mask in the final dataset.', y=0.95)
    for i, (images, masks) in enumerate(supervised_loader):
        image = images[0].permute(1, 2, 0)
        mask = torch.argmax(masks[0], dim=0)

        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Supervised Image")
        axes[0, 1].imshow(mask, cmap='tab20', aspect='equal')
        axes[0, 1].set_title("Supervised Mask")
        fig.delaxes(axes[1, 1])
        break

    # Plot images from the unsupervised loader
    for i, (images, _) in enumerate(unsupervised_loader):
        image = images[0].permute(1, 2, 0)

        axes[1, 0].imshow(image)
        axes[1, 0].set_title("Unsupervised Image")
        break

    # for ax in axes.flat:
    #     ax.axis("off")

    plt.show()

def plot_class_distribution(data_loader: object) -> object:
    """
    Plot the class distribution by pixel share in the dataset.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        DataLoader for the dataset.

    Returns
    -------
    None
    """

    class_counter = torch.zeros(config.NUM_CLASSES)

    for i, (image, mask) in enumerate(data_loader):
        class_counter += torch.bincount(mask[0].view(-1), minlength=config.NUM_CLASSES)

    class_counter = class_counter / class_counter.sum()

    class_names = [config.CLASS_LABELS[i] for i in range(config.NUM_CLASSES)]

    # plt.figure(figsize=(10, 5))
    # plt.bar(class_names, class_counter)
    # plt.xlabel("Class")
    # plt.ylabel("% of Pixels")
    # plt.title("Class Distribution by Pixel Share")
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # plt.show()

    print(class_counter)

def calculate_iou(pred_mask: object, true_mask):
    """
    Calculate Intersection over Union (IoU) between predicted and true masks.

    Parameters
    ----------
    pred_mask : torch.Tensor
        Predicted mask.
    true_mask : torch.Tensor
        True mask.

    Returns
    -------
    iou : float
        Intersection over Union (IoU) score.
    """

    intersection = torch.logical_and(pred_mask, true_mask)
    union = torch.logical_or(pred_mask, true_mask)
    iou = torch.sum(intersection).item() / torch.sum(union).item()
    return iou

def evaluate_model(model, data_loader):
    """
    Evaluate the model's performance using Intersection over Union (IoU) on a dataset.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be evaluated.
    data_loader : torch.utils.data.DataLoader
        DataLoader for the dataset.

    Returns
    -------
    mean_iou : float
        Mean Intersection over Union (IoU) score.
    """

    model.eval()
    iou_score = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            predicted_masks = outputs['out']

            for pred_mask, true_mask in zip(predicted_masks, targets):
                iou_score.append(calculate_iou(pred_mask, true_mask))

    mean_iou = torch.tensor(iou_score).mean().item()
    return mean_iou

def calculate_class_frequencies(mask_path):
    class_frequencies = []
    for root, dirs, files in os.walk(mask_path):
        for img in files:
            mask = rasterio.open(os.path.join(mask_path, img)).read()
            class_counts = np.bincount(mask.flatten(), minlength=16)
            class_frequencies.append(class_counts)

    return np.array(class_frequencies)

def cluster_images(class_frequencies, n_clusters = 6):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(class_frequencies)
    return clusters

def plot_cluster(class_frequencies, clusters):
    tsne = TSNE(n_components=2, random_state=42)
    class_frequencies_tsne = tsne.fit_transform(np.array(class_frequencies))

    pca = PCA(n_components=2)
    class_frequencies_pca = pca.fit_transform(np.array(class_frequencies))

    fig, axes = plt.subplots(2, figsize=(10, 10))
    fig.suptitle('Figure 4. Clusters of Images', y=0.05)

    axes[0].scatter(class_frequencies_tsne[:, 0], class_frequencies_tsne[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    axes[0].set_title("t-SNE Clusters")
    axes[1].scatter(class_frequencies_pca[:, 0], class_frequencies_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    axes[1].set_title("PCA Clusters")

    plt.show()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, preds, labels):
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        labels = labels.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (preds * labels).sum()
        FP = ((1 - labels) * preds).sum()
        FN = (labels * (1 - preds)).sum()

        Tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return 1 - Tversky_index

class ModelWrapper(torch.nn.Module):
    """
    Wrapper class for model with dict/list rvalues.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Init call.
        """
        super().__init__()
        self.model = model

    def forward(self, input_x: torch.Tensor) -> Any:
        """
        Wrap forward call.
        """
        data = self.model(input_x)

        if isinstance(data, dict):
            data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))  # type: ignore
            data = data_named_tuple(**data)  # type: ignore

        elif isinstance(data, list):
            data = tuple(data)

        return data
def wrap_the_model(model):
    return ModelWrapper(model)

def prepare_output_for_calculation(output, target):
    _, predicted = torch.max(output, 1)
    _, true_mask = torch.max(target, 1)
    return predicted, true_mask

def calculate_accuracy(output, target):
    predicted, true_mask = prepare_output_for_calculation(output, target)
    correct = (predicted == true_mask).sum().item()
    total = true_mask.numel()
    return correct / total

def create_experiment_dir(exp_name,  model_name, logs_dir = config.LOG_DIR):
    experiment_path = os.path.join(logs_dir, model_name + '_logs', model_name + exp_name)
    os.makedirs(experiment_path, exist_ok=True)

    return experiment_path

def smooth(scalars: list[float], weight: float) -> list[float]:
    """
    EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed


