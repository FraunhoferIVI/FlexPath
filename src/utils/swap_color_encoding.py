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


def swap_optimal(img, dot_lines: bool = True):
    COLOR_MAPS = [
        ([127, 0, 0], [231, 76, 60]),
        #([0, 127, 0], [45, 85, 45]),
        ([100, 100, 255], [255, 180, 200]),
    ]

    result = swap_color_encoding(img, color_maps=COLOR_MAPS)
    return mark_lines(result, ([231, 76, 60], )) if dot_lines else result # (MIX[1], MAIN_PATH[1]))  # main path and overlap with reference


def swap_pretrained(img, dot_lines: bool = True):
    COLOR_MAPS = [
        ([127, 0, 0], [180, 180, 160]),
        ([0, 127, 0], [45, 85, 45]),
        ([100, 100, 255], [255, 180, 200]),
    ]

    result = swap_color_encoding(img, color_maps=COLOR_MAPS)
    return mark_lines(result, ([180, 180, 160], )) if dot_lines else result # (MIX[1], MAIN_PATH[1]))  # main path and overlap with reference

def swap_waypoint(img, dot_lines: bool = True):
    COLOR_MAPS = [
        ([127, 0, 0], [150, 90, 20]),
        #([0, 127, 0], [45, 85, 45]),
        ([100, 100, 255], [255, 180, 200]),
        ([255, 126, 76], [255, 255, 76])
    ]

    result = swap_color_encoding(img, color_maps=COLOR_MAPS)
    return mark_lines(result, ([150, 90, 20], )) if dot_lines else result # (MIX[1], MAIN_PATH[1]))  # main path and overlap with reference

def swap_obst(img, dot_lines: bool = True):
    COLOR_MAPS = [
        ([127, 0, 0], [26, 188, 156]),
        #([0, 127, 0], [45, 85, 45]),
        ([0, 0, 127], [231, 76, 60]),
        ([100, 100, 255], [255, 180, 200]),
    ]

    result = swap_color_encoding(img, color_maps=COLOR_MAPS)
    return mark_lines(result, ([26, 188, 156], )) if dot_lines else result # (MIX[1], MAIN_PATH[1]))  # main path and overlap with reference


def swap_obs3(img, dot_lines: bool = True):
    COLOR_MAPS = [
        ([127, 0, 0], [52, 152, 219]),
        #([0, 127, 0], [0, 0, 0]), # [25, 45, 25]),
        ([0, 0, 127], [26, 188, 156]),
        ([100, 100, 255], [255, 180, 200]),
    ]

    result = swap_color_encoding(img, color_maps=COLOR_MAPS)
    return mark_lines(result, ([52, 152, 219], )) if dot_lines else result # (MIX[1], MAIN_PATH[1]))  # main path and overlap with reference

def swap_obstlevels(img, dot_lines: bool = True):
    COLOR_MAPS = [
        ([127, 0, 0], [46, 204, 113]),
        #([0, 127, 0], [45, 85, 45]),
        ([0, 0, 127], [26, 188, 156]),
        ([100, 100, 255], [255, 180, 200]),
        ([100, 227, 255], [255, 180, 200]),
    ]

    result = swap_color_encoding(img, color_maps=COLOR_MAPS)
    return mark_lines(result, ([46, 204, 113], )) if dot_lines else result  # (MIX[1], MAIN_PATH[1]))  # main path and overlap with reference


def swap_gt(img, dot_lines: bool = True):
    COLOR_MAPS = [
        ([127, 0, 0], [0, 255, 0]),
        ([0, 127, 0], [25, 45, 25]),
        ([100, 100, 255], [255, 180, 200]),
    ]

    result = swap_color_encoding(img, color_maps=COLOR_MAPS)
    return mark_lines(result, ([0, 255, 0], )) if dot_lines else result  # (MIX[1], MAIN_PATH[1]))  # main path and overlap with reference


def upsample_img(img):
    scale = 16  # 64 * 16 = 1024px (good for slides)
    new_size = (img.width * scale, img.height * scale)

    return img.resize(new_size, Image.Resampling.NEAREST)

def swap_optimal_mod(img, dot_lines: bool = True):
    COLOR_MAPS = [
        ([127, 0, 0], [231, 76, 60]),
        ([20, 80, 20], [0, 127, 0]),
        ([0, 255, 0], [0, 127, 0]),
        ([100, 100, 255], [255, 180, 200]),
    ]

    result = swap_color_encoding(img, color_maps=COLOR_MAPS)
    return result # (MIX[1], MAIN_PATH[1]))  # main path and overlap with reference

def swap_pred(img, dot_lines: bool = True):
    COLOR_MAPS = [
        ([76, 76, 255], [0, 0, 0]),
        ([127, 0, 0], [255, 255, 255]),
        ([255, 255, 76], [0, 0, 0]),
        ([255, 126, 76], [255, 255, 76]),
        ([100, 100, 255], [0, 0, 0]),
    ]

    result = swap_color_encoding(img, color_maps=COLOR_MAPS)
    return result # (MIX[1], MAIN_PATH[1]))  # main path and overlap with reference