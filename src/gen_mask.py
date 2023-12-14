from typing import List
import numpy as np
import cv2
import os


def gen_OTSU_mask(
    img_list: List[str],
    save_dir: str,
):
    """Generate OTSU mask.

    >>> if save_dir != "":
    >>>     os.makedirs(save_dir, exist_ok=True)
    >>>     masks will be saved
    >>>     None will be returned
    >>> else:
    >>>     masks will be returned

    Thumbnail file name is the same as the whole slide image file name.\n
    This function may not perform best on all images, you may need to adjust the code included manually.

    Parameters:
    ----------------------------------------------------------------
        img_list (List[str]): List of image paths.
        save_dir (str): Directory to save masks.

    Returns:
    ----------------------------------------------------------------
        List[numpy.ndarray] | None: List of thumbnails or None.
    """
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)
        for img_path in img_list:
            save_path = os.path.join(
                save_dir, os.path.splitext(img_path.split("/")[-1])[0] + ".png"
            )
            _gen_otsu_mask(img_path, save_path)
    else:
        mask_list = []
        for img_path in img_list:
            mask = _gen_otsu_mask(img_path, "")
            mask_list.append(mask)
        return mask_list


def _gen_otsu_mask(img_path: str, save_path: str):
    img = cv2.imread(img_path)
    thumbnail_blur = cv2.blur(img, (11, 11))
    # H:hue(色相0~359) S:Saturation(饱和度0~1) V:value(明度0~1)
    thumbnail_HSV = cv2.cvtColor(thumbnail_blur, cv2.COLOR_BGR2HSV)
    thumbnail_S = thumbnail_HSV[:, :, 1]
    ret, tissue_mask_grey = cv2.threshold(
        thumbnail_S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    tissue_mask = dilate_and_erode(tissue_mask_grey)
    if save_path != "":
        cv2.imwrite(save_path, tissue_mask)
        return None
    else:
        return tissue_mask


def dilate_and_erode(im: np.ndarray):
    # 膨胀，填充空洞
    kernel = np.ones((8, 8), np.uint8)  # 根据实际情况调整
    im = cv2.dilate(im.astype(np.uint8), kernel, iterations=1)

    # 腐蚀掉白色碎点
    kernel = np.ones((8, 8), np.uint8)  # 根据实际情况调整
    im = cv2.erode(im.astype(np.uint8), kernel, iterations=1)

    # 先识别所有离散的白色区域碎片，通过连通域识别，填补为背景黑色
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        im.astype(np.uint8), connectivity=8
    )
    # print(stats.shape)

    # 通过面积筛选，去除碎片
    for i in range(1, stats.shape[0]):
        if stats[i, cv2.CC_STAT_AREA] < 1000:  # 根据实际情况调整
            im[labels == i] = 0

    # 同样的方法，去除白色中的小的黑色区域
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        1 - im.astype(np.uint8), connectivity=8
    )
    # print(stats.shape)
    for i in range(1, stats.shape[0]):
        # print(stats[i, cv2.CC_STAT_AREA])
        if stats[i, cv2.CC_STAT_AREA] < 1100:  # 根据实际情况调整
            im[labels == i] = 1

    return im
