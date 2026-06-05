# Copyright 2023 Taehyoung Kim
# Maybe apply normalization twice messes up the training.

import os
import pickle
import random

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import distance_transform_edt

from tqdm import tqdm


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


# Calculate reward based on the predefined reward function
def calculate_gt_reward(logits, labels):
    """
    Calculates the ground truth reward function as a target

    Args:
    logits (numpy.array): The input image represented as a Numpy array. [B x H x W]
    labels (numpy.array): The target label represented as a NumPy array. [B x H x W]

    Returns:
    gt_reward (float): A calculate gt_reward based on reward shaping
    """

    # choose action (here we have to change this to stochastic value)
    logits = logits.numpy()  # [B X H x W] to numpy
    labels = labels.numpy()  # [B X H x W] to numpy

    # IsPath:
    # Check if the path is valid (continuous)
    # If not give no reward / if valid give scalar reward

    # Optimality:
    # To calculate the optimality, we can use the length of the reference A* path. A shorter distance would be more optimal. The length of the path can simply calculated
    # by counting the number labels.
    A_star_distance = np.count_nonzero(labels == 2, axis=(1, 2))
    actual_distance = np.count_nonzero(logits == 2, axis=(1, 2))

    # The optimality metric can be calculated as the ratio of the optimal distance and the actual distance of the path. Let's call this ratio opt_ratio.
    # opt_ratio = A*_distance / actual_distance

    opt_ratio = (A_star_distance + 1) / (actual_distance + 1)  # avoid zero division
    opt_ratio[opt_ratio > 1] = 0

    # Since the generated path cannot be optimal than A* path, think of penalizing the case of opt_ratio being larger than 1.

    # Energy efficiency:
    # To calculate the energy efficiency of the path, we can count the number of waypoints in the path. A path with fewer waypoints would be more energy efficient.
    # We can count the number of waypoints by iterating through the path and checking if the direction changes at each step. If the direction changes, we add 1 to the waypoint count.
    # The energy efficiency metric can be calculated as the ratio of the minimum number of waypoints required to traverse the path and the actual number of waypoints in the path.
    # Let's call this ratio efficiency_ratio.

    # efficiency_ratio = actual_waypoint / A*_waypoint

    # Clearance:
    # To calculate the clearance of the path, we can measure the distance between the path and the obstacles. A path that is farther away from the obstacles would have better clearance.
    # We can iterate through the path and calculate the distance between each point on the path and the nearest obstacle. We can take the minimum distance as the clearance of the path.
    # The clearance metric can be calculated as the ratio of the maximum allowed distance and the actual minimum distance between the path and the obstacles. Let's call this ratio clearance_ratio.

    # clearance_ratio = actual_clearance / A*_clearance

    # Now we can take the weighted sum of these metrics to calculate the reward. Let's say the weights of the three metrics are w_opt, w_efficiency, and w_clearance. The reward can be calculated as follows:

    w_opt = 1

    gt_reward = w_opt * opt_ratio  # apply only opt_ratio for the time being
    # + w_efficiency * efficiency_ratio + w_clearance * clearance_ratio

    # We can adjust the weights of the three metrics to emphasize different aspects of the reward. For example, if we want to prioritize optimality over energy efficiency and clearance, we can set w_opt to a higher value than w_efficiency and w_clearance.
    # Optimality based on A* path
    return gt_reward


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


def is_valid_path(start, end, logits, threshold):
    # Set the device to cpu for faster computation
    return_device = start.device  # to gpu when returning the result
    device = "cpu"

    # Start and End to cpu
    start = start.to(device)
    end = end.to(device)

    # predictions
    predictions = (torch.sigmoid(logits) >= threshold).int().to(device)  # to 'cpu'

    B, W, H = predictions.shape
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]
    directions = torch.tensor(directions, device=device)  # Convert directions to tensor

    # Initialize results tensor
    results = torch.zeros(B, dtype=torch.bool, device=device)

    # Loop over each item in the batch
    for i in range(B):
        current = start[i]

        while not torch.all(current == end[i]):
            found_next = False

            for direction in directions:
                new_x, new_y = current[0] + direction[0], current[1] + direction[1]
                if 0 <= new_x < W and 0 <= new_y < H:  # Bound check
                    if predictions[i][new_x][new_y] == 1 or (
                        new_x == end[i][0] and new_y == end[i][1]
                    ):  # Valid path or end point
                        predictions[i][current[0]][current[1]] = 0
                        current = torch.tensor([new_x, new_y], device=device)
                        found_next = True
                        break

            if not found_next:
                results[i] = False  # The path is not valid for this item
                break
        else:
            results[i] = True  # The path is valid for this item

    return results.to(return_device)  # to gpu


def path_optimality(logits, labels, threshold):
    # predictions
    predictions = (torch.sigmoid(logits) >= threshold).int()  # [B x H x W]
    a_star = labels  # [B x H x W]

    count_prediction_path = torch.sum(predictions, dim=(1, 2))  # [B]
    count_a_star_path = torch.sum(a_star == 1, dim=(1, 2))  # [B]

    # count the number of path cells
    optimaility = (1 + count_a_star_path) / (1 + count_prediction_path)  # [B]

    return optimaility


# torch.complie
# @torch.compile
def critic_is_valid_path(start, end, preds):
    return_device = start.device
    device = "cpu"

    start = start.to(device)
    end = end.to(device)
    predictions = preds.to(device)

    B, W, H = predictions.shape
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]
    directions = torch.tensor(directions, device=device)

    results = torch.zeros(B, dtype=torch.bool, device=device)

    # take this backtracking algorithm out // optimize
    def backtrack(current, visited, i):
        if torch.all(current == end[i]):
            # print("Reached end point", current)
            return True

        visited.add(tuple(current.tolist()))

        for direction in directions:
            new_x, new_y = current[0] + direction[0], current[1] + direction[1]
            next_point = torch.tensor([new_x, new_y], device=device)

            if 0 <= new_x < W and 0 <= new_y < H:
                if tuple(next_point.tolist()) not in visited:
                    visited.add(tuple(next_point.tolist()))  # Mark as visited
                    if predictions[i, new_x, new_y] == 1 or torch.all(next_point == end[i]):
                        # print("Moving to:", next_point)
                        if backtrack(next_point, visited, i):
                            return True

        # print("Backtracking from:", current)
        return False

    for i in range(B):
        visited = set()
        start_tuple = tuple(start[i].tolist())  # Convert to Python tuple for set operations
        visited.add(start_tuple)  # Mark the start point as visited
        results[i] = backtrack(start[i], visited, i)

    return results.to(return_device)


def critic_is_collision(initial_obs, pred_path):
    # Assuming initial_obs and pred_path are tensors of shape [batch_size, height, width, channels]
    # Define obstacle color
    obstacle_color = torch.tensor([76 / 255, 76 / 255, 255 / 255]).to(initial_obs.device)  # BLUE

    # Identify obstacles in the unnormalized observation
    # This will give a tensor of shape [batch_size, height, width] with True where there's an obstacle
    obstacle_map = torch.all(initial_obs == obstacle_color[:, None, None], dim=1)

    # Check if there is any overlap between the path and the obstacles
    # Assuming pred_path is a binary mask of shape [batch_size, height, width]
    collision = obstacle_map & pred_path.bool()

    # Check if there's a collision in any of the batches
    return torch.any(torch.any(collision, dim=1), dim=1)

def critic_is_collision_rawdata(initial_obs, pred_path):
    # Assuming initial_obs and pred_path are tensors of shape [batch_size, height, width, channels]
    # Define obstacle color
    obstacle_color = torch.tensor([76, 76, 255]).to(initial_obs.device)  # BLUE

    # Identify obstacles in the unnormalized observation
    # This will give a tensor of shape [batch_size, height, width] with True where there's an obstacle
    obstacle_map = torch.all(initial_obs == obstacle_color[None, None, :], dim=-1)

    # Check if there is any overlap between the path and the obstacles
    # Assuming pred_path is a binary mask of shape [batch_size, height, width]
    collision = obstacle_map & pred_path.bool()

    # Check if there's a collision in any of the batches
    return torch.any(torch.any(collision, dim=1), dim=1)


# this leverages cpu only library
def critic_is_in_vicinity(predicted_path, optimal_path, max_distance=5):
    # Ensure the tensors are on the CPU
    optimal_path = optimal_path.cpu()
    predicted_path = predicted_path.cpu()

    # Placeholder for distance maps
    batch_size, height, width = predicted_path.shape
    distance_maps = torch.empty((batch_size, height, width))

    # Calculate the Euclidean distance transform for each optimal path in the batch
    for i in range(batch_size):
        distance_maps[i] = torch.tensor(distance_transform_edt(1 - optimal_path[i].numpy()))

    # Calculate reward
    reward_map = torch.where(distance_maps <= max_distance, torch.tensor(0), torch.tensor(-1))

    # Reward for predicted path, summed over the spatial dimensions
    reward = torch.sum(reward_map * predicted_path, dim=[1, 2])

    return reward


def count_nearby_obstacles(obs, pred_path, n):
    """
    Count the number of obstacle pixels near path pixels within the distance of n pixels using PyTorch.

    Parameters:
    - obs: 4D tensor representing the batch of unnormalized obs (Batch x Channels x Height x Width).
    - pred_path: 3D tensor representing the predicted paths (Batch x Height x Width).
    - n: distance around the path pixel to check.

    Returns:
    - tensor: total count of obstacle pixels near path pixels for each image in the batch.
    """

    device = obs.device

    # Define obstacle color
    obstacle_color = torch.tensor([76 / 255, 76 / 255, 255 / 255], dtype=torch.float32).view(1, 3, 1, 1).to(device)

    # Create a mask for obstacle pixels
    obstacle_mask = torch.all(obs == obstacle_color, dim=1)

    # Get the coordinates of the path
    path_coords = torch.nonzero(pred_path, as_tuple=True)

    # Tensor to store counts for each batch
    counts = torch.zeros(obs.shape[0], dtype=torch.int, device=device)

    for b in torch.unique(path_coords[0]):
        curr_coords = torch.stack(
            [path_coords[1][path_coords[0] == b], path_coords[2][path_coords[0] == b]],
            dim=1,
        )

        count = torch.tensor(0, device=device, dtype=torch.int)

        for x, y in curr_coords:
            x_min, x_max = max(0, x - n), min(obs.shape[2], x + n + 1)
            y_min, y_max = max(0, y - n), min(obs.shape[3], y + n + 1)

            # Get the subgrid around the path pixel
            subgrid = obstacle_mask[b, x_min:x_max, y_min:y_max]

            # Don't count the center pixel itself if it's also an obstacle
            if x - x_min < subgrid.shape[0] and y - y_min < subgrid.shape[1]:
                subgrid[x - x_min, y - y_min] = False

            # Update the count
            count += torch.sum(subgrid)

        counts[b] = count

    return counts


# Reward condiering action probabilites as well -> did not work out so well
def critic_reward(
    predicted_path_length,
    optimal_path_length,
    predicted_action_path_prob_sum,
    is_valid_path,
    alpha=0.05,
    beta=0.02,
    invalid_path_reward=0.8,
    invalid_path_factor=0.5,
):
    # Device
    device = is_valid_path.device

    # Ensure paths are non-zero to prevent division by zero
    predicted_path_length = predicted_path_length.clamp(min=1)
    optimal_path_length = optimal_path_length.clamp(min=1)
    predicted_action_path_prob_sum = predicted_action_path_prob_sum.clamp(min=1)

    # Penalties for deviation from optimal path length
    length_penalty = (predicted_path_length - optimal_path_length).clamp(max=0)
    extra_length_reward = (predicted_path_length - optimal_path_length).clamp(min=0)

    # Calculate base reward based on path validity
    is_valid_path_reward = torch.where(
        is_valid_path, torch.tensor(1.0).to(device), torch.tensor(invalid_path_reward).to(device)
    )

    # Give higher rewards to agent being more sure about the path
    confidence_ratio = predicted_action_path_prob_sum / predicted_path_length

    # Final reward is a combination of all factors
    # First check if the path is valid, if the path is valid the resulting reward should be 1, otherwise
    reward = (
        confidence_ratio * is_valid_path_reward + alpha * length_penalty
    )  # currently we do not penalize for branches

    # Apply penalty factor to invalid paths + for valid path with extra length
    reward = torch.where(is_valid_path, reward - beta * extra_length_reward, reward * invalid_path_factor)

    # Ensure reward is within [0, 1]
    reward = reward.clamp(min=0, max=1)

    return reward


# Basic reward we used to use, without vicinity and collision
def critic_reward_BR(
    predicted_path_length,
    optimal_path_length,
    is_valid_path,
    alpha=0.05,
    beta=0.02,
    invalid_path_reward=0.8,
    invalid_path_factor=0.5,
):
    # Device
    device = is_valid_path.device

    # Ensure paths are non-zero to prevent division by zero
    predicted_path_length = predicted_path_length.clamp(min=1)
    optimal_path_length = optimal_path_length.clamp(min=1)

    # Penalties for deviation from optimal path length
    length_penalty = (predicted_path_length - optimal_path_length).clamp(max=0)
    extra_length_reward = (predicted_path_length - optimal_path_length).clamp(min=0)

    # Calculate base reward based on path validity
    is_valid_path_reward = torch.where(
        is_valid_path, torch.tensor(1.0).to(device), torch.tensor(invalid_path_reward).to(device)
    )

    # Final reward is a combination of all factors
    # First check if the path is valid, if the path is valid the resulting reward should be 1, otherwise
    reward = is_valid_path_reward + alpha * length_penalty  # currently we do not penalize for branches

    # Apply penalty factor to invalid paths + for valid path with extra length
    reward = torch.where(is_valid_path, reward - beta * extra_length_reward, reward * invalid_path_factor)

    # Ensure reward is within [0, 1]
    reward = reward.clamp(min=0, max=1)

    return reward


# Default reward having vicinity and collision
def critic_reward_DR(
    predicted_path_length,
    optimal_path_length,
    is_valid_path,
    is_in_vicinity,
    is_collision,
    alpha=0.05,
    beta=0.02,
    invalid_path_reward=0.8,
    invalid_path_factor=0.5,
):
    # Device
    device = is_valid_path.device

    # Ensure paths are non-zero to prevent division by zero
    predicted_path_length = predicted_path_length.clamp(min=1)
    optimal_path_length = optimal_path_length.clamp(min=1)

    # Penalties for deviation from optimal path length
    length_penalty = (predicted_path_length - optimal_path_length).clamp(max=0)
    extra_length_reward = (predicted_path_length - optimal_path_length).clamp(min=0)

    # Calculate base reward based on path validity
    is_valid_path_reward = torch.where(
        is_valid_path, torch.tensor(1.0).to(device), torch.tensor(invalid_path_reward).to(device)
    )

    # Final reward is a combination of all factors
    # First check if the path is valid, if the path is valid the resulting reward should be 1, otherwise
    reward = is_valid_path_reward + alpha * length_penalty  # currently we do not penalize for branches

    # Apply penalty factor to invalid paths + for valid path with extra length
    reward = torch.where(is_valid_path, reward - beta * extra_length_reward, reward * invalid_path_factor)

    # Ensure reward is within [0, 1]
    reward = reward.clamp(min=0, max=1)

    # If the path is going haywhere
    # Create a mask for where the conditions are met / consider removing is in vicinity just to give it more freedom of selecting path
    condition_mask = (is_in_vicinity < 0) | is_collision

    # Set reward to zero where the conditions are met
    reward[condition_mask] = 0

    return reward


# Default reward having vicinity and collision
def critic_reward_LR(
    predicted_path_length,
    optimal_path_length,
    is_valid_path,
    is_in_vicinity,
    is_collision,
    alpha=0.005,
    beta=0.01,
):
    # Device
    device = is_valid_path.device

    # Ensure paths are positive values or zero
    predicted_path_length = predicted_path_length.clamp(min=0)
    optimal_path_length = optimal_path_length.clamp(min=0)

    # Penalties for deviation from optimal path length
    extra_length_penalty = (predicted_path_length - optimal_path_length).clamp(min=0)
    extra_length_reward = (predicted_path_length).clamp(min=0)

    # Calculate base reward based on path validity
    is_valid_path_reward = torch.where(is_valid_path, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))

    # Final reward is a combination of all factors
    # First check if the path is valid, if the path is valid the resulting reward should be 1, otherwise
    connected_path_reward = (is_valid_path_reward - extra_length_penalty * beta).clamp(min=0.7, max=1)
    disconnected_path_reward = (is_valid_path_reward + extra_length_reward * alpha).clamp(min=0, max=0.3)

    # Apply extra length reward to invalid paths + for valid path penalize them with extra_length_penalty
    reward = torch.where(is_valid_path, connected_path_reward, disconnected_path_reward)

    # If the path is going haywhere
    # Create a mask for where the conditions are met / is in vicinity will constain its space once again
    condition_mask = (is_in_vicinity < 0) | is_collision

    # Set reward to zero where the conditions are met
    reward[condition_mask] = 0

    return reward


# Default reward having vicinity and collision
def critic_reward_LR_v2(
    predicted_path_length,
    optimal_path_length,
    is_valid_path,
    is_in_vicinity,
    is_collision,
    alpha=0.005,
    beta=0.3,
):
    # Device
    device = is_valid_path.device

    # Ensure paths are positive values or zero
    predicted_path_length = predicted_path_length.clamp(min=1)
    optimal_path_length = optimal_path_length.clamp(min=0)

    # Penalties for deviation from optimal path length
    extra_length_penalty = (optimal_path_length / predicted_path_length).clamp(min=0)
    extra_length_reward = (predicted_path_length).clamp(min=0)

    # Calculate base reward based on path validity
    is_valid_path_reward = torch.where(is_valid_path, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))

    # Final reward is a combination of all factors
    # First check if the path is valid, if the path is valid the resulting reward should be 1, otherwise
    connected_path_reward = (is_valid_path_reward - (1 - extra_length_penalty)).clamp(min=0.5, max=1)
    disconnected_path_reward = (is_valid_path_reward + extra_length_reward * alpha).clamp(min=0, max=0.3)

    # Apply extra length reward to invalid paths + for valid path penalize them with extra_length_penalty
    reward = torch.where(is_valid_path, connected_path_reward, disconnected_path_reward)

    # If the path is going haywhere
    # Create a mask for where the conditions are met / is in vicinity will constain its space once again
    condition_mask = (is_in_vicinity < 0) | is_collision

    # Set reward to zero where the conditions are met
    reward[condition_mask] = 0

    return reward


def critic_reward_LR_v3(
    predicted_path_length,
    optimal_path_length,
    is_valid_path,
    # is_in_vicinity,
    is_collision,
    alpha=0.005,
    beta=0.6,
    gamma=5e-4,
):
    # Device
    device = is_valid_path.device

    # Ensure paths are positive values or zero
    predicted_path_length = predicted_path_length.clamp(min=0)
    optimal_path_length = optimal_path_length.clamp(min=1)

    # Penalties for deviation from optimal path length
    extra_length_penalty = (predicted_path_length / optimal_path_length).clamp(min=0)
    extra_length_reward = (predicted_path_length).clamp(min=0)

    # Calculate base reward based on path validity
    is_valid_path_reward = torch.where(
        is_valid_path, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)
    )

    # Final reward is a combination of all factors
    connected_path_reward = (is_valid_path_reward - beta * torch.abs(1 - extra_length_penalty)).clamp(min=0.5, max=1)
    disconnected_path_reward = (is_valid_path_reward + extra_length_reward * alpha).clamp(min=0, max=0.25)

    # Apply extra length reward to invalid paths + for valid path penalize them with extra_length_penalty
    reward = torch.where(is_valid_path, connected_path_reward, disconnected_path_reward)

    # If the path is going haywire
    condition_mask = is_collision
    reward[condition_mask] = 0

    return reward


def critic_reward_LR_v4(
    predicted_path_length,
    optimal_path_length,
    is_valid_path,
    # is_in_vicinity,
    is_collision,
    nearby_obstacles_count,
    alpha=0.005,
    beta=0.6,
    gamma=5e-4,
):
    # Device
    device = is_valid_path.device

    # Ensure paths are positive values or zero
    predicted_path_length = predicted_path_length.clamp(min=0)
    optimal_path_length = optimal_path_length.clamp(min=1)

    # Penalties for deviation from optimal path length
    extra_length_penalty = (predicted_path_length / optimal_path_length).clamp(min=0)
    extra_length_reward = (predicted_path_length).clamp(min=0)

    # Calculate base reward based on path validity
    is_valid_path_reward = torch.where(
        is_valid_path, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)
    )

    # Final reward is a combination of all factors
    connected_path_reward = (is_valid_path_reward - beta * (1 - extra_length_penalty)).clamp(min=0.5, max=1)
    disconnected_path_reward = (is_valid_path_reward + extra_length_reward * alpha).clamp(min=0, max=0.25)

    # Apply extra length reward to invalid paths + for valid path penalize them with extra_length_penalty
    reward = torch.where(is_valid_path, connected_path_reward, disconnected_path_reward)

    # Add extra obj "nearby_obstacles_count" to the reward
    reward = reward - gamma * nearby_obstacles_count

    # If the path is going haywire
    condition_mask = is_collision
    reward[condition_mask] = 0

    return reward


def precalculate_reward(
    images: torch.Tensor,
    a_star_paths: torch.Tensor,
    pred_paths: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the reward for the given dataset. Only use this if reward is invariant to all augmentations that are applied later on.

    Params:
    - images: The state
    - a_star_paths: An optimal path
    - pred_paths: The predicted path

    Returns:
    - torch.Tensor with a reward for each sample: Shape=[samples, ]

    """

    assert len(images) == len(a_star_paths) == len(pred_paths), "The length of the given arrays do not match."

    # store rewards here
    rewards = list()

    # color maps
    start_color = torch.tensor([255, 76, 76]) # not normalized yet
    end_color = torch.tensor([76, 255, 76])
    
    for image, a_star_path, pred_path in tqdm(zip(images, a_star_paths, pred_paths)):
        # convert to tensor
        image = torch.from_numpy(image)
        a_star_path = torch.from_numpy(a_star_path)
        pred_path = torch.from_numpy(pred_path)

        # Extract start and end coordinates from image
        # Find start and end pixels
        find_start = torch.all(image == start_color.view(1, 1, -1), dim=-1)
        find_end = torch.all(image == end_color.view(1, 1, -1), dim=-1)

        start_indices = torch.nonzero(find_start, as_tuple=False).squeeze()
        end_indices = torch.nonzero(find_end, as_tuple=False).squeeze()

        # Handle edge case: if start/end has multiple pixels, take the first one
        if start_indices.dim() > 1:
            start_indices = start_indices[0]
        if end_indices.dim() > 1:
            end_indices = end_indices[0]

        count_astar_path = torch.count_nonzero(a_star_path == 1, dim=(0, 1))

        # Here we first convert the pred_path to binary / pred_path has float values between -1 and 1
        pred_path_binary = torch.where(pred_path > 0, 1, 0)

        count_pred_path = torch.count_nonzero(pred_path_binary == 1, dim=(0, 1))
        # print(count_pred_path, "pred_path_counter", count_pred_path.shape)

        is_valid_path = critic_is_valid_path(start_indices.unsqueeze(0), end_indices.unsqueeze(0), pred_path_binary.clone().unsqueeze(0))  # bool
        # print(f"is valid path: {is_valid_path, {is_valid_path.dtype}}")
        is_collision = critic_is_collision_rawdata(image.unsqueeze(0), pred_path_binary.unsqueeze(0))
        # print(f"is collision path: {is_collision}, {is_collision.dtype}")

        reward = critic_reward_LR_v3(count_pred_path, count_astar_path, is_valid_path, is_collision)

        rewards.append(reward)

    return torch.stack(rewards)
# def precalculate_reward_from_fastenv(
#     images: torch.Tensor,
#     pred_paths: torch.Tensor,
#     target_paths: torch.Tensor,
#     use_path_as_target: bool = False
# ) -> torch.Tensor:
#     """
#     Calculates the reward for the given dataset. Only use this if reward is invariant to all augmentations that are applied later on.

#     Params:
#     - images: The state
#     - a_star_paths: An optimal path
#     - pred_paths: The predicted path

#     Returns:
#     - torch.Tensor with a reward for each sample: Shape=[samples, ]

#     """

#     assert len(images) == len(target_paths) == len(pred_paths), "The length of the given arrays do not match."

#     # store rewards here
#     rewards = list()

#     end_color = np.array([76, 255, 76])
#     start_color = np.array([255, 76, 76])

#     # create wrapper env
#     env = VMADatasetEnv(cache_reward_components=True)
#     # env.reset()
#     env.chevyshev_distance = False

#     if use_path_as_target:
#         target_paths = pred_paths
    
#     for image, target_path, pred_path in tqdm(zip(images, target_paths, pred_paths)):

#         env.rndint = 0

#         find_start = np.all(image == start_color[:, np.newaxis, np.newaxis], axis=0)
#         find_end = np.all(image == end_color[:, np.newaxis, np.newaxis], axis=0)

#         start = np.nonzero(find_start)
#         end = np.nonzero(find_end)

#         # set values
#         env.pred_path = pred_path
#         env.initial_world = image
#         env.obs_world = image
#         env.targets[0] = np.sum(target_path)
#         env.target_predictions[0] = target_path

#         # set astar path as target too
#         env.astar_path = target_path
        
#         env.start = start
#         env.end = end

#         # calc reward
#         reward = env._compute_reward()

#         rewards.append(reward)

#     rewards = np.stack(rewards)
#     rewards = np.expand_dims(rewards, -1)

#     print("Rewards.shape: ", rewards.shape)
#     print("First 10 rewards: ", rewards[:10])

#     return rewards
