import os
import random
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Helper function to reduce class number
def replace_color(img):
    """
    Vectorized replacement of start and end point colors to white.

    Args:
    - img: A NumPy array representing an RGB image.

    Returns:
    - A NumPy array representing the modified image.
    """
    mask_start = np.all(img == [255, 76, 76], axis=-1)
    mask_end = np.all(img == [76, 255, 76], axis=-1)

    img[mask_start] = [255, 255, 255]
    img[mask_end] = [255, 255, 255]
    return img


# Helper function to convert RGB target to labeled target
def RGB_to_label(img, class_num):
    """
    Replaces all the pixels in color of [R, G, B] with corresponding label read from dictionary.

    Args:
    - img: A NumPy array representing an RGB image. [H x W x C]

    Returns:
    - A NumPy array representing the labeled pixels. [H x W]
    """

    SEG_LABELS_LIST_5_CLASS = [
        {"id": 0, "name": "void", "rgb_values": [0, 0, 0]},
        {"id": 1, "name": "path", "rgb_values": [255, 255, 255]},
        {"id": 2, "name": "obstacle", "rgb_values": [76, 76, 255]},
        {"id": 3, "name": "start", "rgb_values": [255, 76, 76]},
        {"id": 4, "name": "end", "rgb_values": [76, 255, 76]},
    ]

    SEG_LABELS_LIST_3_CLASS = [
        {"id": 0, "name": "void", "rgb_values": [0, 0, 0]},
        {"id": 1, "name": "path", "rgb_values": [255, 255, 255]},
        {"id": 2, "name": "obstacle", "rgb_values": [76, 76, 255]},
    ]

    SEG_LABELS_LIST = SEG_LABELS_LIST_5_CLASS if class_num == 5 else SEG_LABELS_LIST_3_CLASS
    labels_dict = {tuple(x["rgb_values"]): x["id"] for x in SEG_LABELS_LIST}

    labels = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for rgb, label_id in labels_dict.items():
        mask = np.all(img == list(rgb), axis=-1)
        labels[mask] = label_id

    return labels


def generate_image_and_masks(env, num_image_class, image_scale_ratio):
    """
    Generate and return image, labeled RGB mask, path-only mask, labeled image and path-only labeled mask.
    """
    obs, info = env.reset()
    image = np.moveaxis(info["unnormalized obs"], 0, 2)

    if num_image_class == 3:
        image = replace_color(image)

    mask_rgb = image.copy()
    for waypoints in info["astar path"][1:-1]:
        mask_rgb[
            waypoints[1] * image_scale_ratio : waypoints[1] * image_scale_ratio + image_scale_ratio,
            waypoints[0] * image_scale_ratio : waypoints[0] * image_scale_ratio + image_scale_ratio,
            :,
        ] = 255

    path_mask_rgb = np.zeros(image.shape, dtype=image.dtype)
    mask_path = np.all(mask_rgb == [255, 255, 255], axis=-1)
    path_mask_rgb[mask_path] = [255, 255, 255]

    mask = RGB_to_label(mask_rgb, class_num=num_image_class)
    path_mask = RGB_to_label(path_mask_rgb, class_num=num_image_class)

    return image, mask_rgb, mask, path_mask_rgb, path_mask


def save_image(arr, directory, filename):
    """Save a given numpy array as an image in the specified directory."""
    filepath = os.path.join(directory, filename)
    image = Image.fromarray(arr)
    image.save(filepath)


def generate_subplot_actor(p_save_dir, s1_save_dir, s2_save_dir, s3_save_dir, s4_save_dir):
    """
    Generate a subplot of three images from the specified directories.

    Args:
        p_save_dir (str): Path to directory containing original images.
        s1_save_dir (str): Path to directory containing RGB labeled masks.
        s2_save_dir (str): Path to directory containing labeled masks.
        s3_save_dir (str): Path to directory containing RGB labeled path masks.
        s4_save_dir (str): Path to directory containing labeled path masks.

    Returns:
        None. Displays the generated subplot using matplotlib.
    """
    # Get list of file names in directories
    p_files = os.listdir(p_save_dir)
    s1_files = os.listdir(s1_save_dir)
    s2_files = os.listdir(s2_save_dir)
    s3_files = os.listdir(s3_save_dir)
    s4_files = os.listdir(s4_save_dir)

    # Make sure the number of files is the same in each directory
    assert (
        len(p_files) == len(s1_files) == len(s2_files) == len(s3_files) == len(s4_files)
    ), "Number of files must be the same in each directory"

    # Randomly select an index
    index = random.randint(0, len(p_files) - 1)
    print(f"map index of {index} displayed")

    # Get file names at the selected index
    p_file = p_files[index]
    s1_file = s1_files[index]
    s2_file = s2_files[index]
    s3_file = s3_files[index]
    s4_file = s4_files[index]

    # Load images using PIL
    p_image = Image.open(os.path.join(p_save_dir, p_file))
    s1_image = Image.open(os.path.join(s1_save_dir, s1_file))
    s2_image = Image.open(os.path.join(s2_save_dir, s2_file))
    s3_image = Image.open(os.path.join(s3_save_dir, s3_file))
    s4_image = Image.open(os.path.join(s4_save_dir, s4_file))

    # Create subplot
    fig, ax = plt.subplots(1, 5, figsize=(10, 5))
    ax[0].imshow(p_image)
    ax[0].set_title("Image")
    ax[1].imshow(s1_image)
    ax[1].set_title("Mask RGB")
    ax[2].imshow(s2_image)
    ax[2].set_title("Mask Labeled")
    ax[3].imshow(s3_image)
    ax[3].set_title("Mask Path RGB")
    ax[4].imshow(s4_image)
    ax[4].set_title("Mask Path Labeled")

    # Show the subplot
    plt.show()


def generate_subplot_critic(p_save_dir, s1_save_dir, s2_save_dir):
    """
    Generate a subplot of three images from the specified directories.

    Args:
        p_save_dir (str): Path to directory containing original images.
        s1_save_dir (str): Path to directory containing A* path.
        s2_save_dir (str): Path to directory containing predicted path.

    Returns:
        None. Displays the generated subplot using matplotlib.
    """
    # Get list of file names in directories
    p_files = os.listdir(p_save_dir)
    s1_files = os.listdir(s1_save_dir)
    s2_files = os.listdir(s2_save_dir)

    # Make sure the number of files is the same in each directory
    assert len(p_files) == len(s1_files) == len(s2_files), "Number of files must be the same in each directory"

    # Randomly select an indexs
    index = random.randint(0, len(p_files) - 1)
    print(f"map index of {index} displayed")

    # Get file names at the selected index
    p_file = p_files[index]
    s1_file = s1_files[index]
    s2_file = s2_files[index]

    # Load images using PIL
    p_image = Image.open(os.path.join(p_save_dir, p_file))
    s1_image = Image.open(os.path.join(s1_save_dir, s1_file))
    s2_image = np.load(os.path.join(s2_save_dir, s2_file))

    # Create subplot
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(p_image)
    ax[0].set_title("Image")
    ax[1].imshow(s1_image)
    ax[1].set_title("Path Astar")
    ax[2].imshow(s2_image, cmap="gray")
    ax[2].set_title("Path Prediction")

    # Show the subplot
    plt.show()


def load_images_from_hf_dataset(dataset, split: str) -> Dict:
    """Loads images from a given hf dataset split."""
    images = {
        "image": dataset[split]["image"],
        "label": dataset[split]["label"],
        "rgb_label": dataset[split]["rgb_label"],
        "path_label": dataset[split]["path_label"],
        "path_rgb_label": dataset[split]["path_rgb_label"],
    }
    return images


def load_images_from_hf_dataset_critic(dataset, split: str) -> Dict:
    """Loads images from a given hf dataset split."""
    images = {
        "image": dataset[split]["image"],
        "astar_path": dataset[split]["astar_path"],
        "pred_path": dataset[split]["pred_path"],
    }
    return images


def ensure_dir_exists(dir_path):
    """Ensure that the given directory exists. If not, create it."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
