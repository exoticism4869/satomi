from multiprocessing import Pool
from openslide import OpenSlide
from satomi.kfb.KfbSlide import KfbSlide
from satomi.wsi2thumb import wsi2thumb
from tqdm import tqdm
import numpy as np
import cv2
import os
import random


def crop_patch(
    wsi_path_list,
    mask_dir,
    color,
    save_dir,
    patch_size,
    crop_downsample,
    mask_downsample,
    stride,
    threshold,
    max_patch_per_wsi=0,
    num_workers=10,
):
    """Crop patches from wsi

    Patches on the explicit level will be resized from level0 version patch.\n
    The patches that cropped from the same wsi will be saved in the same directory named by wsi id.\n
    The patch will be named by {wsi_id}_{topleft_i}_{topleft_j}.png

    Parameters:
    ----------------------------------------------------------------
        wsi_path_list (List[str]): List of wsi paths.
        mask_dir (str): Directory of masks.
        color(List[int]): Color to choose of the mask, e.g. [255, 255, 255] for white.
        save_dir (str): Directory to save patches.
        patch_size (int): Patch size on the desired level.
        crop_downsample (int): Downsample factor for cropping, 1=level0, 2=level1.
        mask_downsample (int): Downsample factor for masks.
        stride (int): Distance that shift patch moves, the number is relative to patch_size. e.g.(patch_size=256, stride=224)
        threshold (float): Accept patch if the proportion of white area in the patch bigger than threshold.
        max_patch_per_wsi (int): The most patch number that can be accepted from one wsi, 0 means no limit.
        num_workers (int): Number of workers.

    Returns:
    ----------------------------------------------------------------
        None
    """
    task = []
    for wsi_path in wsi_path_list:
        wsi_id = os.path.splitext(wsi_path.split("/")[-1])[0]
        mask_path = os.path.join(mask_dir, wsi_id + ".png")
        save_dir_now = os.path.join(save_dir, wsi_id)
        task.append(
            (
                wsi_path,
                mask_path,
                color,
                save_dir_now,
                patch_size,
                crop_downsample,
                mask_downsample,
                stride,
                threshold,
                max_patch_per_wsi,
            )
        )
    pool = Pool(num_workers)
    with tqdm(total=len(task), desc="Cropping patch") as pbar:
        for _ in pool.imap(_crop_patch, task):
            pbar.update(1)
    pool.close()
    pool.join()


def _gen_topleft_points(
    mask_path, color, patch_size, crop_downsample, mask_downsample, stride, threshold
):
    # color主要针对有多种颜色的mask，比如有些mask是有多个类别的，这时候需要指定颜色
    patch_size_level0 = patch_size * crop_downsample
    mask_patch_size = patch_size_level0 // mask_downsample
    stride = stride * crop_downsample // mask_downsample
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_height, mask_width, _ = mask.shape
    topleft = []  # 计算有效patch在level0的左上角坐标
    for i in range(0, mask_height, stride):
        for j in range(0, mask_width, stride):
            if j + stride > mask_width or i + stride > mask_height:
                continue
            temp = mask[i : i + mask_patch_size, j : j + mask_patch_size]
            if (np.all(temp == color, axis=2)).sum() / (
                mask_patch_size**2
            ) > threshold:
                topleft_i = int(
                    (i + mask_patch_size / 2) * mask_downsample
                    - patch_size * crop_downsample / 2
                )
                topleft_j = int(
                    (j + mask_patch_size / 2) * mask_downsample
                    - patch_size * crop_downsample / 2
                )
                topleft.append((topleft_i, topleft_j))
    return topleft


def _crop_patch(params):
    (
        wsi_path,
        mask_path,
        color,
        save_dir,
        patch_size,
        crop_downsample,
        mask_downsample,
        stride,
        threshold,
        max_patch_per_wsi,
    ) = params
    patch_size_level0 = patch_size * crop_downsample
    topleft = _gen_topleft_points(
        mask_path,
        color,
        patch_size,
        crop_downsample,
        mask_downsample,
        stride,
        threshold,
    )
    if len(topleft) == 0:
        return
    os.makedirs(save_dir, exist_ok=True)
    if max_patch_per_wsi != 0:
        topleft = random.sample(topleft, min(len(topleft), max_patch_per_wsi))
    wsi_id, extension = os.path.splitext(wsi_path.split("/")[-1])
    if extension == ".kfb":
        slide = KfbSlide(wsi_path)
    else:
        slide = OpenSlide(wsi_path)
    for topleft_i, topleft_j in topleft:
        patch = slide.read_region(
            (topleft_j, topleft_i), 0, (patch_size_level0, patch_size_level0)
        )
        patch = patch.resize((patch_size, patch_size))
        save_path = os.path.join(
            save_dir, wsi_id + "_" + str(topleft_i) + "_" + str(topleft_j) + ".png"
        )
        patch.save(save_path)


def crop_vis(
    thumb_path_list,
    wsi_path_list,
    mask_dir,
    color,
    save_dir,
    patch_size,
    crop_downsample,
    mask_downsample,
    stride,
    threshold,
):
    """Visualize crop region on thumbnail

    Either wsi_path_list or thumb_path_list must be provided.\n

    Parameters:
    ----------------------------------------------------------------
        thumb_path_list(List[str]): List of thumbnail paths, thumbnail downsample must be equal to mask_downsample.
        wsi_path_list (List[str]): List of wsi paths.
        mask_dir (str): Directory of masks.
        color(List[int]): Color to choose of the mask, e.g. [255, 255, 255] for white.
        save_dir (str): Directory to save visualized thumbnails.
        patch_size (int): Patch size on the desired level.
        crop_downsample (int): Downsample factor for cropping, 1=level0, 2=level1.
        mask_downsample (int): Downsample factor for masks.
        stride (int): Distance that shift patch moves, the number is relative to patch_size. e.g.(patch_size=256, stride=224)
        threshold (float): Accept patch if the proportion of white area in the patch bigger than threshold.

    Returns:
    ----------------------------------------------------------------
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    mask_patch_size = patch_size * crop_downsample // mask_downsample
    if len(thumb_path_list) != 0:
        for thumb_path in thumb_path_list:
            wsi_id = os.path.splitext(thumb_path.split("/")[-1])[0]
            mask_path = os.path.join(mask_dir, wsi_id + ".png")
            save_path = os.path.join(save_dir, wsi_id + ".png")
            topleft = _gen_topleft_points(
                mask_path,
                color,
                patch_size,
                crop_downsample,
                mask_downsample,
                stride,
                threshold,
            )
            if len(topleft) == 0:
                continue
            thumb = cv2.imread(thumb_path)
            for topleft_i, topleft_j in topleft:
                cv2.rectangle(
                    thumb,
                    (topleft_j // mask_downsample, topleft_i // mask_downsample),
                    (
                        topleft_j // mask_downsample + mask_patch_size,
                        topleft_i // mask_downsample + mask_patch_size,
                    ),
                    (0, 0, 255),
                    thickness=2,
                )
            cv2.imwrite(save_path, thumb)
    elif len(wsi_path_list) != 0:
        for wsi_path in wsi_path_list:
            wsi_id = os.path.splitext(wsi_path.split("/")[-1])[0]
            mask_path = os.path.join(mask_dir, wsi_id + ".png")
            save_path = os.path.join(save_dir, wsi_id + ".png")
            topleft = _gen_topleft_points(
                mask_path,
                color,
                patch_size,
                crop_downsample,
                mask_downsample,
                stride,
                threshold,
            )
            if len(topleft) == 0:
                continue
            (thumb,) = wsi2thumb([wsi_path], save_dir="", downsample=mask_downsample)
            thumb = np.array(thumb)
            thumb = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
            for topleft_i, topleft_j in topleft:
                cv2.rectangle(
                    thumb,
                    (topleft_j // mask_downsample, topleft_i // mask_downsample),
                    (
                        topleft_j // mask_downsample + mask_patch_size,
                        topleft_i // mask_downsample + mask_patch_size,
                    ),
                    (0, 0, 255),
                    thickness=2,
                )
            cv2.imwrite(save_path, thumb)
    else:
        raise ValueError("Either wsi_path_list or thumb_path_list must be provided.")


def crop_vis_from_patch(
    patch_dir,
    patch_size,
    save_dir,
    wsi_dir,
    thumb_dir,
    crop_downsample,
    thumb_downsample,
    point_position="topleft",
):
    """Visualize crop region on thumbnail from already cropped patch

    For already cropped patch, the patch name must be {wsi_id}_{topleft_i}_{topleft_j}.png\n
    Either wsi_dir or thumb_dir must be provided.\n

    Parameters:
    ----------------------------------------------------------------
        patch_dir (str): Directory of patch, the last part of this path must be wsi id.
        patch_size (int): Patch size.
        save_dir (str): Directory to save visualized thumbnails.
        wsi_dir (str): Directory of wsi.
        thumb_dir (str): Directory of thumbnail.
        crop_downsample (int): Downsample factor for cropping, 1=level0, 2=level1.
        thumb_downsample (int): Downsample factor for thumbnail.When wsi_dir is provided, thumb_downsample means the downsample factor for thumbnail, when thumb_dir is provided, thumb_downsample means the downsample factor for exsited thumbnail.
        point_position (str): The position of the point, "topleft" or "center".

    Returns:
    ----------------------------------------------------------------
        None
    """

    def find_wsi_path(tiff_dir, wsi_id):
        for cur_dir, dirs, files in os.walk(tiff_dir):
            for file in files:
                if wsi_id == os.path.splitext(file)[0]:
                    return os.path.join(cur_dir, file)

    os.makedirs(save_dir, exist_ok=True)
    if thumb_dir != "":
        pass

    elif wsi_dir != "":
        wsi_id = patch_dir.split("/")[-1]
        wsi_path = find_wsi_path(wsi_dir, wsi_id)
        thumb = wsi2thumb([wsi_path], save_dir="", downsample=thumb_downsample)[0]
        thumb = np.array(thumb)
        thumb = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
        for patch_name in os.listdir(patch_dir):
            patch_id = os.path.splitext(patch_name)[0]
            wsi_id, topleft_i, topleft_j = patch_id.split("_")
            if point_position == "center":
                topleft_i = int(topleft_i) - patch_size * crop_downsample // 2
                topleft_j = int(topleft_j) - patch_size * crop_downsample // 2
            elif point_position == "topleft":
                topleft_i = int(topleft_i)
                topleft_j = int(topleft_j)
            else:
                raise ValueError("point_position must be topleft or center.")
            cv2.rectangle(
                thumb,
                (
                    topleft_j // thumb_downsample,
                    topleft_i // thumb_downsample,
                ),
                (
                    topleft_j // thumb_downsample
                    + patch_size * crop_downsample // thumb_downsample,
                    topleft_i // thumb_downsample
                    + patch_size * crop_downsample // thumb_downsample,
                ),
                (0, 0, 255),
                thickness=2,
            )
        save_path = os.path.join(save_dir, wsi_id + ".png")
        cv2.imwrite(save_path, thumb)
    else:
        raise ValueError("Either wsi_dir or thumb_dir must be provided.")


def dataset_split(
    wsi_path_list,
    ratio,
    mask_dir,
    color,
    patch_size,
    crop_downsample,
    mask_downsample,
    stride,
    threshold,
    max_patch_per_wsi=0,
):
    """Split the dataset into train and test dataset.

    Parameters
    ----------------------------------------------------------------
    ratio (float): The ratio of train dataset.

    Returns
    ----------------------------------------------------------------
    train_path (List[str]): List of train wsi path.
    test_path (List[str]): List of test wsi path.
    """

    patch_num = []  # 记录每个wsi可切的patch数量
    for wsi_path in wsi_path_list:
        wsi_id = os.path.splitext(wsi_path.split("/")[-1])[0]
        mask_path = os.path.join(mask_dir, wsi_id + ".png")
        topleft = _gen_topleft_points(
            mask_path,
            color,
            patch_size,
            crop_downsample,
            mask_downsample,
            stride,
            threshold,
        )
        if max_patch_per_wsi != 0:
            patch_num.append(min(len(topleft), max_patch_per_wsi))
        else:
            patch_num.append(len(topleft))
    patch_num = np.array(patch_num)
    patch_sum = patch_num.sum()

    test_num = int(patch_sum * (1 - ratio))
    test_path = []
    # 随机从wsi_list中选取wsi，直到test_num<0
    order = list(range(len(wsi_path_list)))
    random.shuffle(order)
    for i in order:
        if patch_num[i] > test_num:
            break
        else:
            test_num -= patch_num[i]
            test_path.append(wsi_path_list[i])
    train_path = list(set(wsi_path_list) - set(test_path))
    return train_path, test_path


if __name__ == "__main__":
    wsi_path = "/data/lymphnode/tiff/泛癌淋巴结/4肺癌-淋巴结/0112318-OK/0112318D.kfb"
    wsi_path2 = "/data/cesc/tiff/20221115/tif/王改霞/王改霞（1）0001659468.tif"
    mask_dir = "/home/hdc/Sebs/nnunet_otsu_min_fillmask"
    save_dir = "/home/hdc/Sebs/test/patch"
    patch_size = 512
    crop_downsample = 2
    mask_downsample = 40
    stride = 512
    threshold = 0.9
    crop_patch(
        [wsi_path],
        mask_dir,
        save_dir,
        patch_size,
        crop_downsample,
        mask_downsample,
        stride,
        threshold,
    )
