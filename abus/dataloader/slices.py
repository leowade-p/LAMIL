# data_loader.py - ABUS 2D 切片数据加载
import glob
import logging
import os

import pandas as pd


def load_slice_data(split_name: str, png_data_root: str, csv_path: str) -> list:
    """
    扫描预处理 PNG 目录，结合 CSV 标签，返回切片级数据列表。

    每个元素: {'image_path', 'mask_path', 'pid', 'label'}
    """
    logging.info(f"Loading 2D slice data for '{split_name}' split...")

    if not os.path.exists(csv_path):
        logging.error(f"Label CSV file not found at: {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    label_map = {
        str(row["case_id"]): 1 if str(row["label"]).upper() in ["M", "1"] else 0
        for _, row in df.iterrows()
    }

    image_dir = os.path.join(png_data_root, split_name.lower(), "images")
    mask_dir = os.path.join(png_data_root, split_name.lower(), "masks")

    if not os.path.exists(image_dir):
        logging.error(f"Image directory not found at: {image_dir}")
        return []

    slice_list = []
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))

    for img_path in image_files:
        filename = os.path.basename(img_path)
        try:
            parts = filename.split("_")
            pid = parts[1]
            mask_filename = filename.replace(".png", "_mask.png")
            mask_path = os.path.join(mask_dir, mask_filename)

            if not os.path.exists(mask_path):
                continue

            patient_label = label_map.get(pid)
            if patient_label is None:
                continue

            slice_list.append(
                {
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "pid": pid,
                    "label": patient_label,
                }
            )
        except IndexError:
            logging.warning(f"Could not parse filename '{filename}', skipping.")
            continue

    logging.info(f"Found {len(slice_list)} slices for '{split_name}' split.")
    return slice_list
