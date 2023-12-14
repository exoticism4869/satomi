from typing import List, Tuple
import numpy as np
import cv2
import collections
import json
import os


def json2mask(
    json_list: List[str],
    downsample: int,
    color_list: List[str],
    mask_size: Tuple[int],
    save_dir: str,
):
    """Convert json to mask.

    >>> if save_dir != "":
    >>>     os.makedirs(save_dir, exist_ok=True)
    >>>     masks will be saved
    >>>     None will be returned
    >>> elif len(json_list) > 10:
    >>>     raise ValueError
    >>> else:
    >>>     masks will be returned

    Thumbnail file name is the same as the whole slide image file name.

    Parameters:
    ----------------------------------------------------------------
        json_list (List[str]): List of json paths.
        downsample (int): Downsample factor according to the mask size you choose. Json coordinates are based on level 0 WSI, so they need to be transformed.
        color_list (List[str]): List of color e.g. ['#4a90e2', '#f8e71c'], indicating colors used to draw mask.
        mask_size (Tuple[int]): Size (width, height) of mask, usually size is same to thumbnail size.
        save_dir (str): Directory to save masks.

    Returns:
    ----------------------------------------------------------------
        List[numpy.ndarray] | None: List of thumbnails or None.
    """
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)
        for json_path in json_list:
            save_path = os.path.join(
                save_dir, os.path.splitext(json_path.split("/")[-1])[0] + ".png"
            )
            _gen_otsu_mask(json_path, downsample, color_list, mask_size, save_path)
    elif len(json_list) > 10:
        raise ValueError(
            "The length of json_list more than 10, save_dir must be specified."
        )
    else:
        mask_list = []
        for json_path in json_list:
            mask = _gen_otsu_mask(json_path, downsample, color_list, mask_size, "")
            mask_list.append(mask)
        return mask_list


def _gen_otsu_mask(
    json_path: str,
    downsample: int,
    color_list: List[str],
    mask_size: Tuple[int],
    save_path: str,
):
    with open(json_path, "r") as json_data:
        contour_list = json.load(json_data)
    contour_dict = collections.defaultdict(list)

    for contour in contour_list:
        color = contour["color"]
        if color not in color_list:
            continue
        # json文件中coordinates为三维列表，所以取[0]
        points = np.array(contour["coordinates"][0])
        points = (points / downsample).astype(int)
        contour_dict[color].append(points)
    mask_black = np.zeros(mask_size, dtype="uint8")
    for hex, contours in contour_dict.items():
        cv2.drawContours(mask_black, contours, -1, 255, -1)
    if save_path != "":
        cv2.imwrite(save_path, mask_black)
        return None
    else:
        return mask_black
