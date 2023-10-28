import numpy as np
import rasterio
from rasterio.plot import show
from rasterio.plot import reshape_as_raster, reshape_as_image
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from patchify import patchify,unpatchify
import os
from glob import glob
from PIL import Image
import config

def cut_image(image_name, n=config.PATCH_SIZE, m=config.PATCH_SIZE):
    """
    Cut an input image into sub-images of specified dimensions.

    Parameters
    ----------
    image_name : numpy.ndarray
        The input image to be cut.
    n : int, optional
        Width of the sub-images. Defaults to the value in config.PATCH_SIZE.
    m : int, optional
        Height of the sub-images. Defaults to the value in config.PATCH_SIZE.

    Returns
    -------
    sub_images : list of numpy.ndarray
        List of sub-images.
    """

    return [image_name[x:x + m, y:y + n] for x in range(0, image_name.shape[0], m) for y in
            range(0, image_name.shape[1], n)]

def random_index(arr):
    """
    Get a random index from an array.

    Parameters
    ----------
    arr : list or numpy.ndarray
        The input array.

    Returns
    -------
    int
        A random index from the array.
    """

    return np.random.randint(0, len(arr) - 1)


def meets_criteria(sub_image, threshold=0.9):
    """
    Check if a sub-image meets criteria for a threshold for non-background pixels.

    Parameters
    ----------
    sub_image : numpy.ndarray
        The sub-image to be checked.
    threshold : float, optional
        The threshold for non-background pixels. Defaults to 0.9.

    Returns
    -------
    bool
        True if the criteria are met, False otherwise.
    """

    unique, counts = np.unique(sub_image,
                               return_counts=True)
    perc = counts[-1] / np.sum(counts)

    # 255 was used because most of the incomplete images have white spaces
    if unique[-1] != 255 or perc < 0.9:
        return True
    else:
        return False

def process_image(region, image_path, results_path, mask_path = False, image_size = config.ORIGINAL_IMAGE_SIZE):
    """
    Process an image, potentially with an associated mask, by cutting it into sub-images and saving them.

    Parameters
    ----------
    region : str
        The region or category of the image.
    image_path : str
        Path to the input image.
    results_path : str
        Path to the directory where sub-images will be saved.
    mask_path : str or False, optional
        Path to the mask image (if available). Defaults to False.
    image_size : int, optional
        Size of the original image. Defaults to the value in 'config.ORIGINAL_IMAGE_SIZE'.

    Returns
    -------
    None
    """

    image = rasterio.open(image_path)
    mask = rasterio.open(mask_path) if mask_path else None

    if image.width == image.height == image_size:
        image_data = reshape_as_image(image.read())
        sub_images = cut_image(image_data)

        criteria_met = False
        while not criteria_met:
            sub_tile_idx = random_index(sub_images)
            criteria_met = meets_criteria(sub_images[sub_tile_idx])

        sub_image = Image.fromarray(sub_images[sub_tile_idx])
        image_code = os.path.basename(image_path)[8:17]
        sub_image.save(os.path.join(results_path, f'{region}_{image_code}_{sub_tile_idx}.tif'))

        if mask and mask.width == mask.height == image_size:
            mask_data = reshape_as_image(mask.read())
            sub_masks = cut_image(mask_data)
            sub_mask = Image.fromarray(np.squeeze(sub_masks[sub_tile_idx], axis=2))
            sub_mask.save(os.path.join(results_path, 'masks', f'mask_{region}_{image_code}_{sub_tile_idx}.tif'))

def data_preparation(regions = config.REGIONS, data_path = config.ORIGINAL_DATA_DIR, results_path = config.DATA_DIR, max_images = config.MAX_IMAGES):
    """
    Prepare data by cutting images into sub-images and saving them in specified directories.

    Parameters
    ----------
    regions : list, optional
        List of regions or categories. Defaults to 'config.REGIONS'.
    data_path : str, optional
        Path to the original data directory. Defaults to 'config.ORIGINAL_DATA_DIR'.
    results_path : str, optional
        Path to the directory where results will be saved. Defaults to 'config.DATA_DIR'.
    MAX_IMAGES : int, optional
        Maximum number of images to process. Defaults to 1000.

    Returns
    -------
    None
    """

    used_images = []

    for region in regions:
        region_path = os.path.join(data_path, region)
        print(f'Region: {region}, Number of Satellite images: {len(os.listdir(region_path))}')
        for img in os.listdir(region_path):
            image_path = os.path.join(region_path, img)
            mask_path = os.path.join(data_path, 'labels', region, img[:-8] + '_UA2012.tif')

            if os.path.exists(mask_path):
                # Process the image with an associated mask
                process_image(region, image_path, results_path, mask_path)

            else:
                # Process the image without an associated mask
                process_image(region, image_path, results_path)

            used_images.append(image_path)

    print('Second run')

    # Perform secondary run on the original images, to fill the max_image requirement
    while len(used_images) < max_images:
        code = regions[random_index(regions)]
        images = os.listdir(os.path.join(data_path, code))
        img = images[random_index(images)]
        image_path = os.path.join(data_path, code, img)
        mask_path = os.path.join(data_path, 'labels', code, img[:-8] + '_UA2012.tif')

        if os.path.exists(mask_path):
            # Process the image with an associated mask
            process_image(code, image_path, results_path, mask_path)

        else:
            # Process the image without an associated mask
            process_image(code, image_path, results_path)

        used_images.append(image_path)

if __name__ == "__main__":
    data_preparation()