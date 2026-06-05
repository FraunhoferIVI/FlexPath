import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from PIL import Image

import cv2


def swap_color_encoding(
    source_img: np.ndarray,
    color_maps: tuple,
) -> np.ndarray:
    """
    Docstring for swap_color_encoding
    
    :param source_img: np array of image to be transformed: [H, W, 3]
    :type source_img: np.ndarray
    :param color_maps: Tuple of (<color_encoding_source>, <desired_color_encoding>)
    :type color_maps: dict
    :return: Description
    :rtype: ndarray
    """

    result_img = source_img.copy()

    for i in range(len(color_maps)):
        current_color_encoding = np.array(color_maps[i][0])
        desired_color_encoding = np.array(color_maps[i][1])

        # compute mask of pixels to change
        change_mask = np.logical_and(np.logical_and(source_img[:, :, 0] == current_color_encoding[0], source_img[:, :, 1] == current_color_encoding[1]), source_img[:, :, 2] == current_color_encoding[2])
        print(change_mask)
        result_img[change_mask, 0] = desired_color_encoding[0]
        result_img[change_mask, 1] = desired_color_encoding[1]
        result_img[change_mask, 2] = desired_color_encoding[2]
    
    return result_img

def mark_lines(
    source_img: np.ndarray,
    color_encodings_to_mark: tuple,
) -> np.ndarray:
    upsampled = cv2.resize(
        source_img,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_NEAREST
    )

    for color_encoding_to_mark in color_encodings_to_mark:
        pixels_to_mark = np.argwhere(np.all(source_img == color_encoding_to_mark, axis=-1))
        print(pixels_to_mark.shape)
        pixels_to_mark_new_scale = (3 * pixels_to_mark) + 1
        upsampled[pixels_to_mark_new_scale[:, 0], pixels_to_mark_new_scale[:, 1]] = np.array([255, 255, 255])

    return upsampled

# Current
# Obstacle: [76, 76, 255]
# Empty: [0, 0, 0]
# Start&End: [255, 255, 76]
# Path: [127, 0, 0]
# Path on obstacle: [203, 76, 255]
# Expanded region: [0, 127, 0]
# Waypoint: [255, 255, 76]
# Waypoint on expanded region: 

"""
([127, 0, 0], [255, 0, 0]),
        ([0, 127, 0], [0, 255, 0]),
        ([0, 0, 0], [255, 255, 255]),
        ([76, 76, 255], [0, 0, 0]),
"""

"""
([0, 0, 0], [16, 16, 24]),
([76, 76, 255], [155, 89, 182]),
"""

def swap_optimal(img, dot_lines: bool = True):
    COLOR_MAPS = [
        ([127, 0, 0], [231, 76, 60]),
        ([0, 127, 0], [0, 127, 0]),
        ([100, 100, 255], [255, 180, 200]),
    ]

    result = swap_color_encoding(img, color_maps=COLOR_MAPS)
    return mark_lines(result, ([231, 76, 60], )) if dot_lines else result # (MIX[1], MAIN_PATH[1]))  # main path and overlap with reference


def swap_pretrained(img, dot_lines: bool = True):
    COLOR_MAPS = [
        ([127, 0, 0], [180, 180, 160]),
        ([0, 127, 0], [25, 45, 25]),
        ([100, 100, 255], [255, 180, 200]),
    ]

    result = swap_color_encoding(img, color_maps=COLOR_MAPS)
    return mark_lines(result, ([180, 180, 160], )) if dot_lines else result # (MIX[1], MAIN_PATH[1]))  # main path and overlap with reference

def swap_waypoint(img, dot_lines: bool = True):
    COLOR_MAPS = [
        ([127, 0, 0], [150, 90, 20]),
        ([0, 127, 0], [0, 0, 0]),
        ([100, 100, 255], [255, 180, 200]),
    ]

    result = swap_color_encoding(img, color_maps=COLOR_MAPS)
    return mark_lines(result, ([150, 90, 20], )) if dot_lines else result # (MIX[1], MAIN_PATH[1]))  # main path and overlap with reference

def swap_obst(img, dot_lines: bool = True):
    COLOR_MAPS = [
        ([127, 0, 0], [26, 188, 156]),
        ([0, 127, 0], [25, 45, 25]),
        ([0, 0, 127], [231, 76, 60]),
        ([127, 0, 127], [128, 116, 121]),
        ([100, 100, 255], [255, 180, 200]),
    ]

    result = swap_color_encoding(img, color_maps=COLOR_MAPS)
    return mark_lines(result, ([26, 188, 156], [128, 116, 121])) if dot_lines else result # (MIX[1], MAIN_PATH[1]))  # main path and overlap with reference


def swap_obs3(img, dot_lines: bool = True):
    COLOR_MAPS = [
        ([127, 0, 0], [52, 152, 219]),
        ([0, 127, 0], [25, 45, 25]),
        ([0, 0, 127], [26, 188, 156]),
        ([127, 0, 127], [39, 170, 171]),
        ([100, 100, 255], [255, 180, 200]),
    ]

    result = swap_color_encoding(img, color_maps=COLOR_MAPS)
    return mark_lines(result, ([52, 152, 219], [39, 170, 171])) if dot_lines else result # (MIX[1], MAIN_PATH[1]))  # main path and overlap with reference

def swap_obstlevels(img, dot_lines: bool = True):
    COLOR_MAPS = [
        ([127, 0, 0], [46, 204, 113]),
        ([0, 127, 0], [25, 45, 25]),
        ([0, 0, 127], [26, 188, 156]),
        ([127, 0, 127], [36, 196, 135]),
        ([100, 100, 255], [255, 180, 200]),
    ]

    result = swap_color_encoding(img, color_maps=COLOR_MAPS)
    return mark_lines(result, ([46, 204, 113], [36, 196, 135])) if dot_lines else result  # (MIX[1], MAIN_PATH[1]))  # main path and overlap with reference


def swap_gt(img, dot_lines: bool = True):
    COLOR_MAPS = [
        ([127, 0, 0], [0, 255, 0]),
        ([0, 127, 0], [25, 45, 25]),
        ([100, 100, 255], [255, 180, 200]),
    ]

    result = swap_color_encoding(img, color_maps=COLOR_MAPS)
    return mark_lines(result, ([0, 255, 0], )) if dot_lines else result  # (MIX[1], MAIN_PATH[1]))  # main path and overlap with reference

def recover(img):
    COLOR_MAPS = [
        ([255, 255, 255], [26, 188, 156]), 
        ([128, 116, 121], [255, 255, 255])
    ]

    result = swap_color_encoding(img, color_maps=COLOR_MAPS)
    return result  # (MIX[1], MAIN_PATH[1]))  # main path and overlap with reference


def revert(img):
    COLOR_MAPS = [
        ([255, 0, 0], [0, 0, 0]), 
        ([128, 116, 121], [255, 255, 255])
    ]

    result = swap_color_encoding(img, color_maps=COLOR_MAPS)
    return result  # (MIX[1], MAIN_PATH[1]))  # main path and overlap with reference

if __name__ == '__main__':
    SRC_PATH = "selected_images/optimality_ood_colormapped/voxelgym"
    # TARGET_PATH = "selected_images/front_comparison/Y/_gt.png"  # "test.png"  # "voxelgym_figureimages/sample_0000152_hardness_1.2971_baseimg.png"  # SRC_PATH  # "swaptest.png"
    files = [f for f in Path(SRC_PATH).iterdir()
         if f.is_file() and f.suffix.lower() == ".png"]

    for file in files:
    # COLOR_MAPS = [
    #     ([100, 100, 255], [76, 76, 255]),
    # ]
    # # OVERLAY = ([0, 0, 127], [231, 76, 60])
    # MAIN_PATH = ([127, 0, 0], [231, 76, 60])  # [241, 196, 15])

    # COLOR_MAPS.append(MAIN_PATH)
    # COLOR_MAPS.append(OVERLAY)
    # MIX = ([127, 0, 127], [(MAIN_PATH[1][0] + OVERLAY[1][0]) // 2, (MAIN_PATH[1][1] + OVERLAY[1][1]) // 2, (MAIN_PATH[1][2] + OVERLAY[1][2]) // 2])
    # COLOR_MAPS.append(MIX)

        img = np.asarray(Image.open(file))

        result = swap_optimal(img, dot_lines=False)

        # result = mark_lines(result, (MAIN_PATH[1], )) # (MIX[1], MAIN_PATH[1]))  # main path and overlap with reference
        result = Image.fromarray(result)# .save(file)  # "test.png")

        scale = 16  # 64 * 16 = 1024px
        new_size = (result.width * scale, result.height * scale)

        result = result.resize(new_size, Image.Resampling.NEAREST)
        result.save(file)