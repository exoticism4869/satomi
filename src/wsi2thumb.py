from multiprocessing import Pool
from openslide import OpenSlide
from satomi.kfb.KfbSlide import KfbSlide
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List
import torch
import pyvips
import numpy as np
import cv2
import os


def wsi2thumb(
    wsi_path_list: List[str],
    save_dir: str,
    downsample: int,
    num_workers: int = 10,
):
    """Convert whole slide images to thumbnails.

    >>> if save_dir != "":
    >>>     os.makedirs(save_dir, exist_ok=True)
    >>>     thumbnails will be saved
    >>>     None will be returned
    >>> else:
    >>>     thumbnails will be returned

    Thumbnail file name is the same as the whole slide image file name.

    Parameters:
    ----------------------------------------------------------------
        wsi_path_list (List[str]): List of whole slide image paths.
        save_dir (str): Directory to save thumbnails.
        downsample (int): Downsample factor.
        num_workers (int): Number of workers.

    Returns:
    ----------------------------------------------------------------
        list[PIL.Image] | None: List of thumbnails or None.
    """

    if save_dir == "":
        thumb_list = []
        for wsi_path in tqdm(wsi_path_list):
            thumb = _gen_thumb((wsi_path, "", downsample))
            thumb_list.append(thumb)
        return thumb_list
    else:
        os.makedirs(save_dir, exist_ok=True)
        task = [
            (
                wsi_path,
                os.path.join(
                    save_dir,
                    os.path.splitext(wsi_path.split("/")[-1])[0] + ".png",  # 与wsi同名
                ),
                downsample,
            )
            for wsi_path in wsi_path_list
        ]
        pool = Pool(num_workers)
        with tqdm(total=len(task), desc="Generating thumbnails") as pbar:
            for _ in pool.imap(_gen_thumb, task):
                pbar.update(1)
        pool.close()
        pool.join()


def _gen_thumb(params):
    wsi_path, save_path, downsample = params
    if os.path.splitext(wsi_path)[1] == ".kfb":
        slide = KfbSlide(wsi_path)
    else:
        slide = OpenSlide(wsi_path)
    thumb = slide.get_thumbnail(
        (slide.dimensions[0] // downsample, slide.dimensions[1] // downsample)
    )
    if save_path != "":
        thumb.save(save_path)
    return thumb


if __name__ == "__main__":
    kfb_path = "/data/lymphnode/tiff/泛癌淋巴结/4肺癌-淋巴结/0112318-OK/0112318D.kfb"
    wsi2thumb([kfb_path], "/home/hdc/Sebs/satomi", 40)
