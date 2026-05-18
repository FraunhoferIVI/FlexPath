import os
import pickle
import random

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2



class CriticTrainTransform:
    def __init__(self, IMAGE_HEIGHT, IMAGE_WIDTH, NORM_PARAMS):
        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        self.IMAGE_WIDTH = IMAGE_WIDTH
        self.NORM_PARAMS = NORM_PARAMS

    def __call__(self, image, mask, mask1):
        # Define the transformations which should be applied to all
        transform_all = A.Compose(
            [
                A.Resize(height=self.IMAGE_HEIGHT, width=self.IMAGE_WIDTH),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ],
            additional_targets={"mask": "mask", "mask1": "mask"},
        )

        # Define the transformation to normalize
        transform_normalize = A.Compose(
            [
                A.Normalize(
                    mean=self.NORM_PARAMS[0], std=self.NORM_PARAMS[1], max_pixel_value=255.0, always_apply=True, p=1.0
                )
            ],
            additional_targets={"mask": "mask"},
        )

        # Transformation to convert to Tensor
        transform_to_tensor = A.Compose([ToTensorV2()], additional_targets={"mask": "mask"})

        # Apply the transformations to all
        transformed = transform_all(image=image, mask=mask, mask1=mask1)

        # Normalize only the image and mask1
        normalized = transform_normalize(image=transformed["image"], mask=transformed["mask"])

        # Convert the normalized image and mask1 to tensors
        tensor_normalized = transform_to_tensor(image=normalized["image"], mask=normalized["mask"])

        # Convert the (non-normalized) transformed mask2 to tensor
        tensor_mask1 = torch.from_numpy(transformed["mask1"])

        # Update the image, mask1 and mask2 in the transformed dict with the tensor versions
        transformed.update(
            {"image": tensor_normalized["image"], "mask": tensor_normalized["mask"], "mask1": tensor_mask1}
        )

        return transformed


class CriticValTransform:
    def __init__(self, IMAGE_HEIGHT, IMAGE_WIDTH, NORM_PARAMS):
        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        self.IMAGE_WIDTH = IMAGE_WIDTH
        self.NORM_PARAMS = NORM_PARAMS

    def __call__(self, image, mask, mask1):
        # Define the transformations which should be applied to all
        transform_all = A.Compose(
            [
                A.Resize(height=self.IMAGE_HEIGHT, width=self.IMAGE_WIDTH),
            ],
            additional_targets={"mask": "mask", "mask1": "mask"},
        )

        # Define the transformation to normalize
        transform_normalize = A.Compose(
            [
                A.Normalize(
                    mean=self.NORM_PARAMS[0], std=self.NORM_PARAMS[1], max_pixel_value=255.0, always_apply=True, p=1.0
                )
            ],
            additional_targets={"mask": "mask"},
        )

        # Transformation to convert to Tensor
        transform_to_tensor = A.Compose([ToTensorV2()], additional_targets={"mask": "mask"})

        # Apply the transformations to all
        transformed = transform_all(image=image, mask=mask, mask1=mask1)

        # Normalize only the image and mask1
        normalized = transform_normalize(image=transformed["image"], mask=transformed["mask"])

        # Convert the normalized image and mask1 to tensors
        tensor_normalized = transform_to_tensor(image=normalized["image"], mask=normalized["mask"])

        # Convert the (non-normalized) transformed mask2 to tensor
        tensor_mask1 = torch.from_numpy(transformed["mask1"])

        # Update the image, mask1 and mask2 in the transformed dict with the tensor versions
        transformed.update(
            {"image": tensor_normalized["image"], "mask": tensor_normalized["mask"], "mask1": tensor_mask1}
        )

        return transformed


obs_means = (24.22 / 255.0, 24.22 / 255.0, 80.36 / 255.0)
obs_stds = (34.18 / 255.0, 34.18 / 255.0, 112.22 / 255.0)


def get_obs_means():
    """voxelgym obs_mean RGB values."""
    return (24.22, 24.22, 80.36)


def get_obs_stds():
    """voxelgym obs_std RGB values."""
    return (34.18, 34.18, 112.22)


def get_sf_train_transform(IMAGE_HEIGHT, IMAGE_WIDTH):
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ]
    )
    return train_transform


def get_unet_train_transform(IMAGE_HEIGHT, IMAGE_WIDTH, NORM_PARAMS):
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=NORM_PARAMS[0], std=NORM_PARAMS[1], max_pixel_value=255.0, always_apply=True, p=1.0),
            ToTensorV2(),
        ]
    )
    return train_transform


def get_sf_val_transform(IMAGE_HEIGHT, IMAGE_WIDTH):
    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        ]
    )
    return val_transform


def get_unet_val_transform(IMAGE_HEIGHT, IMAGE_WIDTH, NORM_PARAMS):
    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=NORM_PARAMS[0], std=NORM_PARAMS[1], max_pixel_value=255.0, always_apply=True, p=1.0),
            ToTensorV2(),
        ]
    )
    return val_transform


def save_checkpoint(state, hp_log, checkpoint_dir, filename="checkpoint.pth.tar"):
    filename = hp_log + filename
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filename = os.path.join(checkpoint_dir, filename)
    torch.save(state, filename, pickle_module=pickle)
    print("=> Saving checkpoint")


def save_actor_checkpoint(state, hp_log, checkpoint_dir, filename="actor_checkpoint.pth.tar"):
    filename = hp_log + filename
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filename = os.path.join(checkpoint_dir, filename)
    torch.save(state, filename, pickle_module=pickle)
    print("=> Saving actor checkpoint")


def save_critic_checkpoint(state, hp_log, checkpoint_dir, filename="critic_checkpoint.pth.tar"):
    filename = hp_log + filename
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filename = os.path.join(checkpoint_dir, filename)
    torch.save(state, filename, pickle_module=pickle)
    print("=> Saving critic checkpoint")


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def voxelgym_3c_palette():
    """voxelgym palette that maps each 3 class to RGB values."""
    return [[0, 0, 0], [255, 255, 255], [76, 76, 255]]


def voxelgym_5c_palette():
    """voxelgym palette that maps each 5 class to RGB values."""
    return [[0, 0, 0], [255, 255, 255], [76, 76, 255], [255, 76, 76], [76, 255, 76], [128, 128, 128]]


def shift_image(input_image, target_label, p=0.0):
    """
    Shifts the image and fills the newly created area with obstacles.

    Args:
    input_image (numpy.array): The input image represented as a Numpy array.
    target_label (numpy.array): The target label represented as a NumPy array.
    p (float): The probablity of the transformation being performed. Assuming that the validation set always see the non-shifted image, we set 0.2 as a default value

    Returns:
    shifted_target (numpy.array): The shifted target with the newly created area filled with obstacles.
    shifted_input (numpy.array): The shifted input with the newly created area filled with obstacles.
    """

    rnd_float = random.random()

    if rnd_float >= p:
        shifted_input = input_image
        shifted_target = target_label

    else:
        # Find out number of classes of input and define path label
        if len(np.unique(target_label)) == 3:  # 3c
            path_label = 1
        if len(np.unique(target_label)) == 5:  # 5c
            path_label = 1
        else:
            path_label = 1

        # Find the coordinates of the start, end and path pixels.
        start_coord = np.argwhere(np.all(input_image == [255, 76, 76], axis=2))
        end_coord = np.argwhere(np.all(input_image == [76, 255, 76], axis=2))
        path_coords = np.argwhere((target_label == path_label))

        # Concantinate all pixels to form path_coords
        path_coords = np.concatenate((start_coord, path_coords, end_coord))

        # Calculate the bounds for random shift values.
        # Calculate the bounds for random shift values.
        min_x = np.min(path_coords[:, 0])
        min_y = np.min(path_coords[:, 1])

        max_x = target_label.shape[0] - 1 - np.max(path_coords[:, 0])
        max_y = target_label.shape[0] - 1 - np.max(path_coords[:, 1])

        # Generate random shift values within the calculated bounds.
        shift_y = random.randint(
            -min_x,
            max_x,
        )
        shift_x = random.randint(-min_y, max_y)

        # Define a transformation matrix for the given shift values.
        shift_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        # Shift the image using the transformation matrix.
        shifted_target = cv2.warpAffine(target_label, shift_matrix, (target_label.shape[1], target_label.shape[0]))
        shifted_input = cv2.warpAffine(input_image, shift_matrix, (input_image.shape[1], input_image.shape[0]))

        # Define the obstacle color.
        obstacle_color = np.array([76, 76, 255], dtype=np.uint8)
        obstacle_label = np.array(0, dtype=np.uint8)  # to void for the moment

        # Fill the newly created area with obstacles.
        if shift_x > 0:
            shifted_target[:, :shift_x] = obstacle_label
            shifted_input[:, :shift_x] = obstacle_color

        elif shift_x < 0:
            shifted_target[:, shift_x:] = obstacle_label
            shifted_input[:, shift_x:] = obstacle_color

        if shift_y > 0:
            shifted_target[:shift_y, :] = obstacle_label
            shifted_input[:shift_y, :] = obstacle_color
        elif shift_y < 0:
            shifted_target[shift_y:, :] = obstacle_label
            shifted_input[shift_y:, :] = obstacle_color

    return shifted_input, shifted_target


def shift_image_and_paths(input_image, astar_path, pred_path, p=0.0):
    """
    Shifts the image and fills the newly created area with obstacles.

    Args:
    input_image (numpy.array): The input image represented as a Numpy array.
    astar_path (numpy.array): The astar path represented as a NumPy array.
    pred_path (numpy.array): The prediction path represented as NuPy array.
    p (float): The probablity of the transformation being performed. Assuming that the validation set always see the non-shifted image, we set 0.5 as a default value

    Returns:
    shifted_input (numpy.array): The shifted input with the newly created area filled with obstacles.
    shifted_astar_path (numpy.array): The shifted astar path with the newly created area filled with obstacles.
    shifted_pred_path (numpy.array): The shifted pred path with the newly created area filled with obstacles.
    """

    rnd_float = random.random()

    if rnd_float >= p:
        shifted_input = input_image
        shifted_astar_path = astar_path
        shifted_pred_path = pred_path

    else:
        path_label = 1

        # Find the coordinates of the start, end and path pixels.
        start_coord = np.argwhere(np.all(input_image == [255, 76, 76], axis=2))
        end_coord = np.argwhere(np.all(input_image == [76, 255, 76], axis=2))
        path_coords = np.argwhere((astar_path == path_label))

        # Concantinate all pixels to form path_coords
        path_coords = np.concatenate((start_coord, path_coords, end_coord))

        # Calculate the bounds for random shift values.
        # Calculate the bounds for random shift values.
        min_x = np.min(path_coords[:, 0])
        min_y = np.min(path_coords[:, 1])

        max_x = astar_path.shape[0] - 1 - np.max(path_coords[:, 0])
        max_y = astar_path.shape[0] - 1 - np.max(path_coords[:, 1])

        # Generate random shift values within the calculated bounds.
        shift_y = random.randint(
            -min_x,
            max_x,
        )
        shift_x = random.randint(-min_y, max_y)

        # Define a transformation matrix for the given shift values.
        shift_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        # Shift the image using the transformation matrix.
        shifted_astar_path = cv2.warpAffine(astar_path, shift_matrix, (astar_path.shape[1], astar_path.shape[0]))
        shifted_pred_path = cv2.warpAffine(pred_path, shift_matrix, (pred_path.shape[1], pred_path.shape[0]))
        shifted_input = cv2.warpAffine(input_image, shift_matrix, (input_image.shape[1], input_image.shape[0]))

        # Define the obstacle color.
        obstacle_color = np.array([76, 76, 255], dtype=np.uint8)
        obstacle_label = np.array(0, dtype=np.uint8)  # to void for the moment

        # Fill the newly created area with obstacles.
        if shift_x > 0:
            shifted_astar_path[:, :shift_x] = obstacle_label
            shifted_pred_path[:, :shift_x] = obstacle_label
            shifted_input[:, :shift_x] = obstacle_color

        elif shift_x < 0:
            shifted_astar_path[:, shift_x:] = obstacle_label
            shifted_pred_path[:, shift_x:] = obstacle_label
            shifted_input[:, shift_x:] = obstacle_color

        if shift_y > 0:
            shifted_astar_path[:shift_y, :] = obstacle_label
            shifted_pred_path[:shift_y, :] = obstacle_label
            shifted_input[:shift_y, :] = obstacle_color
        elif shift_y < 0:
            shifted_astar_path[shift_y:, :] = obstacle_label
            shifted_pred_path[shift_y:, :] = obstacle_label
            shifted_input[shift_y:, :] = obstacle_color

    return shifted_input, shifted_astar_path, shifted_pred_path


# Extract the numeric part of the directory name by reading from right to left
def dataset_size_extracter(dir):
    directory_number = ""
    for character in reversed(dir):
        if character == "_":
            break
        directory_number = character + directory_number

    directory_number = int(directory_number)

    return directory_number


# Float to scientific string
def float_to_scientific_string(number):
    return f"{number:.1e}"


# Seed everything for reproducibility
def seed_everything(seed=24022022):
    """Seed everything for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def show_grid_with_logits(color_seg, logits):
    # Get the dimensions of the original image
    height, width, _ = color_seg.shape

    # Define the grid dimensions
    grid_width = 42  # Number of columns
    grid_height = 42  # Number of rows

    # Compute the size of each grid cell
    cell_width = width // grid_width
    cell_height = height // grid_height

    # Create a grid to store the colors and coordinates
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    # Iterate over the grid cells
    for i in range(grid_height):
        for j in range(grid_width):
            # Compute the coordinates of the grid cell
            x_start = j * cell_width
            y_start = i * cell_height
            x_end = x_start + cell_width
            y_end = y_start + cell_height

            # Compute the color of the grid cell
            cell_color = np.mean(color_seg[y_start:y_end, x_start:x_end], axis=(0, 1))

            # Assign the color to the grid cell
            grid[i, j] = cell_color

            # Assign the logits to the grid cell
            logits_extracted = logits[0, :, i, j]

            # Add logit annotation with the coordinates, upto two decimal places
            logit_text = f"({logits_extracted[0]:.2f}, {logits_extracted[1]:.2f},{logits_extracted[2]:.2f})"
            plt.text(
                x_start + cell_width // 2,
                y_start + cell_height // 2,
                logit_text,
                color="white",
                ha="center",
                va="center",
                fontsize=6,
            )

    # Display the grid
    plt.imshow(grid)
    plt.axis("off")
    plt.show()


def show_grid_with_logits_PO(color_seg, logits):
    # Get the dimensions of the original image
    height, width, _ = color_seg.shape

    # Define the grid dimensions
    grid_width = 42  # Number of columns
    grid_height = 42  # Number of rows

    # Compute the size of each grid cell
    cell_width = width // grid_width
    cell_height = height // grid_height

    # Create a grid to store the colors and coordinates
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    plt.figure(figsize=(15, 15))

    # Iterate over the grid cells
    for i in range(grid_height):
        for j in range(grid_width):
            # Compute the coordinates of the grid cell
            x_start = j * cell_width
            y_start = i * cell_height
            x_end = x_start + cell_width
            y_end = y_start + cell_height

            # Compute the color of the grid cell
            cell_color = np.mean(color_seg[y_start:y_end, x_start:x_end], axis=(0, 1))

            # Assign the color to the grid cell
            grid[i, j] = cell_color

            # Assign the logits to the grid cell
            logits_extracted = logits[0, i, j]

            # Add logit annotation with the coordinates, upto two decimal places
            logit_text = f"({logits_extracted.item():.2f})"  # tensor.item() to convert 0-dim tensor to a number
            plt.text(
                x_start + cell_width // 2,
                y_start + cell_height // 2,
                logit_text,
                color="red",
                ha="center",
                va="center",
                fontsize=6,
            )

    # Display the grid
    plt.imshow(grid)
    plt.axis("off")
    plt.show()


def show_grid_with_logits_PO_NP(color_seg, logits, name=None):
    # Get the dimensions of the original image
    height, width, _ = color_seg.shape

    # Define the grid dimensions
    grid_width = 42  # Number of columns
    grid_height = 42  # Number of rows

    # Compute the size of each grid cell
    cell_width = width // grid_width
    cell_height = height // grid_height

    # Create a grid to store the colors and coordinates
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    plt.figure(figsize=(15, 15))

    # Iterate over the grid cells
    for i in range(grid_height):
        for j in range(grid_width):
            # Compute the coordinates of the grid cell
            x_start = j * cell_width
            y_start = i * cell_height
            x_end = x_start + cell_width
            y_end = y_start + cell_height

            # Compute the color of the grid cell
            cell_color = np.mean(color_seg[y_start:y_end, x_start:x_end], axis=(0, 1))

            # Assign the color to the grid cell
            grid[i, j] = cell_color

            # Assign the logits to the grid cell
            logits_extracted = logits[i, j]

            # Add logit annotation with the coordinates, upto two decimal places
            logit_text = f"({logits_extracted.item():.2f})"  # tensor.item() to convert 0-dim tensor to a number
            plt.text(
                x_start + cell_width // 2,
                y_start + cell_height // 2,
                logit_text,
                color="red",
                ha="center",
                va="center",
                fontsize=6,
            )

    # Display the grid
    plt.imshow(grid)
    plt.axis("off")

    # Save the figure before showing it
    if name is not None:
        plt.savefig(name + ".png", bbox_inches="tight", pad_inches=0)
        # plt.show()
    else:
        plt.show()


def return_grid_with_logits_PO(color_seg, logits):
    height, width, _ = color_seg.shape

    # Define the grid dimensions
    grid_width = 42  # Number of columns
    grid_height = 42  # Number of rows

    # Compute the size of each grid cell
    cell_width = width // grid_width
    cell_height = height // grid_height

    # Create a grid to store the colors and coordinates
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    plt.figure(figsize=(30, 30))

    # Iterate over the grid cells
    for i in range(grid_height):
        for j in range(grid_width):
            # Compute the coordinates of the grid cell
            x_start = j * cell_width
            y_start = i * cell_height
            x_end = x_start + cell_width
            y_end = y_start + cell_height

            # Compute the color of the grid cell
            cell_color = np.mean(color_seg[y_start:y_end, x_start:x_end], axis=(0, 1))

            # Assign the color to the grid cell
            grid[i, j] = cell_color

            # Assign the logits to the grid cell
            logits_extracted = logits[i, j]

            # Add logit annotation with the coordinates, upto two decimal places
            logit_text = f"({logits_extracted:.2f})"  # tensor.item() to convert 0-dim tensor to a number
            plt.text(
                x_start + cell_width // 2,
                y_start + cell_height // 2,
                logit_text,
                color="red",
                ha="center",
                va="center",
                fontsize=8.0,
            )

    # Display the grid
    plt.imshow(grid)
    plt.axis("off")

    fig = plt.gcf()
    plt.close(fig)

    # Return the figure object
    return fig
