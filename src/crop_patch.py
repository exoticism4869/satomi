from multiprocessing import Pool
from openslide import OpenSlide
from satomi.kfb.KfbSlide import KfbSlide
from satomi.wsi2thumb import wsi2thumb
from tqdm import tqdm
import numpy as np
import cv2
import os


def crop_patch(
    wsi_path_list,
    mask_dir,
    save_dir,
    patch_size,
    crop_downsample,
    mask_downsample,
    stride,
    threshold,
    num_workers=10,
):
    """Crop patches from wsi

    Patches on the explicit level will be resized from level0 version patch.\n
    The patches that cropped from the same wsi will be saved in the same directory named by wsi id.\n
    The patch will be named by {wsi_id}_{topleft_i}_{topleft_j}.png

    Parameters:
    ----------------------------------------------------------------
        wsi_path_list (List[str]): List of whole slide image paths.
        mask_dir (str): Directory of masks.
        save_dir (str): Directory to save patches.
        patch_size (int): Patch size on the desired level.
        crop_downsample (int): Downsample factor for cropping, 1=level0, 2=level1.
        mask_downsample (int): Downsample factor for masks.
        stride (int): Distance that shift patch moves, the number is relative to patch_size. e.g.(patch_size=256, stride=224)
        threshold (float): Accept patch if the proportion of white area in the patch bigger than threshold.
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
                save_dir_now,
                patch_size,
                crop_downsample,
                mask_downsample,
                stride,
                threshold,
            )
        )
    pool = Pool(num_workers)
    with tqdm(total=len(task), desc="Cropping patch") as pbar:
        for _ in pool.imap(_crop_patch, task):
            pbar.update(1)
    pool.close()
    pool.join()


def _crop_patch(params):
    (
        wsi_path,
        mask_path,
        save_dir,
        patch_size,
        crop_downsample,
        mask_downsample,
        stride,
        threshold,
    ) = params
    os.makedirs(save_dir, exist_ok=True)
    patch_size_level0 = patch_size * crop_downsample
    mask_patch_size = patch_size_level0 // mask_downsample
    stride = stride * crop_downsample // mask_downsample
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_height, mask_width = mask.shape

    topleft = []  # 计算有效patch在level0的左上角坐标
    for i in range(0, mask_height, stride):
        for j in range(0, mask_width, stride):
            if j + stride > mask_width or i + stride > mask_height:
                continue
            temp = mask[i : i + mask_patch_size, j : j + mask_patch_size]
            if (temp == 255).sum() / (mask_patch_size**2) >= threshold:
                topleft_i = int(
                    (i + mask_patch_size / 2) * mask_downsample
                    - patch_size * crop_downsample / 2
                )
                topleft_j = int(
                    (j + mask_patch_size / 2) * mask_downsample
                    - patch_size * crop_downsample / 2
                )
                topleft.append((topleft_i, topleft_j))

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
