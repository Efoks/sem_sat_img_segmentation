import os
import numpy as np
from PIL import Image
import rasterio
from rasterio.plot import reshape_as_image
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

def meets_criteria(sub_image, sub_mask = None, threshold=0.5):
    """
    Check if a sub-image meets criteria for a threshold for non-background pixels.

    Parameters
    ----------
    sub_image : numpy.ndarray
        The sub-image to be checked.
    threshold : float, optional
        The threshold for non-background pixels. Defaults to 0.5.

    Returns
    -------
    bool
        True if the criteria are met, False otherwise.
    """

    unique, counts = np.unique(sub_image,
                               return_counts=True)
    perc = counts[-1] / np.sum(counts)

    if sub_mask is not None:
        if np.any(sub_mask == 0):
            return False

    # 255 was used because most of the incomplete images have white spaces
    if unique[-1] != 255 or perc < threshold:
        return True
    else:
        return False

def create_directories(base_path):
    supervised_path = os.path.join(base_path, 'supervised_dataset')
    unsupervised_path = os.path.join(base_path, 'unsupervised_dataset')
    supervised_images_path = os.path.join(supervised_path, 'supervised_images')
    masks_path = os.path.join(supervised_path, 'masks')

    os.makedirs(supervised_images_path, exist_ok=True)
    os.makedirs(masks_path, exist_ok=True)
    os.makedirs(unsupervised_path, exist_ok=True)

    return supervised_images_path, masks_path, unsupervised_path

def save_image_mask(sub_image, sub_mask, image_code, sub_tile_idx, region, supervised_images_path, unsupervised_images_path, masks_path):
    img_path = supervised_images_path if sub_mask is not None else unsupervised_images_path
    sub_image_path = os.path.join(img_path, f'{region}_{image_code}_{sub_tile_idx}.tif')
    sub_image.save(sub_image_path)

    if sub_mask is not None:
        sub_mask_path = os.path.join(masks_path, f'mask_{region}_{image_code}_{sub_tile_idx}.tif')
        sub_mask.save(sub_mask_path)

def process_image(region, image_path, supervised_images_path, masks_path, unsupervised_path, mask_path=None, image_size=config.ORIGINAL_IMAGE_SIZE, first_run = True):
    try:
        with rasterio.open(image_path) as image:
            if image.width == image.height == image_size:
                image_data = reshape_as_image(image.read())
                sub_images = cut_image(image_data)

                if mask_path:
                    with rasterio.open(mask_path) as mask:
                        if mask.width == mask.height == image_size:
                            mask_data = reshape_as_image(mask.read())
                            sub_masks = cut_image(mask_data)
                        else:
                            raise ValueError("Mask size doesn't match.")
                else:
                    sub_masks = [None] * len(sub_images)  # No masks available

                criteria_met = False
                attempt_count = 0
                while not criteria_met and attempt_count < 100:
                    idx = np.random.randint(0, 99)
                    sub_image = sub_images[idx]
                    sub_mask = sub_masks[idx]
                    criteria_met = meets_criteria(sub_image, sub_mask)
                    if criteria_met and first_run:
                        save_image_mask(Image.fromarray(sub_image),
                                        Image.fromarray(np.squeeze(sub_mask, axis=2)) if sub_mask is not None else None,
                                        os.path.basename(image_path)[8:17],
                                        idx,
                                        region,
                                        supervised_images_path,
                                        unsupervised_path,
                                        masks_path)
                    attempt_count += 1

                if attempt_count >= 100:
                    print(f"No suitable sub-images found in {image_path}. Moving to next image.")

    except Exception as e:
        print(f"An error occurred while processing the image: {image_path}")
        print(str(e))

def select_random_subset(image_paths, max_images):
    if len(image_paths) > max_images:
        return np.random.choice(image_paths, size=max_images, replace=False)
    return image_paths

def data_preparation(regions=config.REGIONS, data_path=config.ORIGINAL_DATA_DIR, results_path=config.DATA_DIR_V2, max_images = config.MAX_IMAGES):
    supervised_images_path, masks_path, unsupervised_path = create_directories(results_path)

    # Get all image paths from the regions specified
    all_image_paths = [os.path.join(data_path, region, img).replace('/', '\\')
                       for region in regions
                       for img in os.listdir(os.path.join(data_path, region))]

    # Dictionary to hold the mask paths if they exist
    all_mask_paths = {}
    for img_path in all_image_paths:
        mask_path = img_path.split(os.sep)
        mask_path.insert(2, 'labels')
        mask_path[4] = os.path.splitext(mask_path[4])[0].replace('.jp2', '_UA2012.tif')
        mask_path = os.sep.join(mask_path)

        all_mask_paths[img_path] = mask_path if os.path.exists(mask_path) else None

    processed_images = 0
    processed_supervised_images = 0


    # Process each image
    for image_path in all_image_paths:
        region = os.path.basename(os.path.dirname(image_path))
        mask_path = all_mask_paths.get(image_path)
        if mask_path:
            process_image(region, image_path, supervised_images_path, masks_path, unsupervised_path,
                          mask_path=mask_path)
            processed_supervised_images += 1
        else:
            process_image(region, image_path, supervised_images_path, masks_path, unsupervised_path)
        processed_images += 1

    while processed_images < max_images:
        image_path = np.random.choice(all_image_paths)
        region = os.path.basename(os.path.dirname(image_path))
        mask_path = all_mask_paths.get(image_path)

        # Process the image and mask if it exists
        if mask_path and os.path.exists(mask_path):
            process_image(region, image_path, supervised_images_path, masks_path, unsupervised_path, mask_path=mask_path)
            processed_supervised_images += 1
        else:
            process_image(region, image_path, supervised_images_path, None, unsupervised_path)
        processed_images += 1
        print(processed_images)

    print(f'Processed images: {processed_images}\n Supervised image count:{processed_supervised_images}')

if __name__ == "__main__":
    data_preparation()