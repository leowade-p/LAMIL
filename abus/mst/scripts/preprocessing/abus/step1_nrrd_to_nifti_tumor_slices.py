"""
ABUS 预处理：NRRD -> 含瘤 slice stack -> NIfTI (MST 输入格式)

目录结构 (每个 split):
  {Split}/DATA/DATA_{case_id}.nrrd
  {Split}/MASK/MASK_{case_id}.nrrd
  {Split}/labels.csv          # case_id, label (M/B)

输出:
  {output_root}/imagesTr/ABUS_{case_id}_0000.nii.gz
  {output_root}/classification_labels.json
  {output_root}/split.json    # train / val / test keys
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


SPLIT_DIRS = {
    "train": "Train",
    "val": "Validation",
    "test": "Test",
}


def parse_label(raw) -> int:
    return 1 if str(raw).upper() in ("M", "1", "MALIGNANT") else 0


def normalize_case_id(raw) -> str:
    """CSV 里 case_id 可能是 0.0 / 100.0 等浮点，需规范成 '0' / '100'。"""
    if pd.isna(raw):
        return ""
    if isinstance(raw, (int, np.integer)):
        return str(int(raw))
    if isinstance(raw, (float, np.floating)):
        if float(raw).is_integer():
            return str(int(raw))
        return str(raw).strip()
    text = str(raw).strip()
    if text.endswith(".0") and text.replace(".0", "").isdigit():
        return text[:-2]
    return text


def case_id_variants(case_id: str):
    """生成常见文件名变体：0 -> ['0','00','000']，100 -> ['100']。"""
    variants = [case_id]
    if case_id.isdigit():
        num = int(case_id)
        variants.extend([str(num), f"{num:02d}", f"{num:03d}", f"{num:04d}"])
    # 去重且保持顺序
    seen = set()
    out = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def build_file_index(folder: Path, prefix: str) -> dict:
    """扫描 DATA/MASK 目录，建立 {数字id: Path} 索引。"""
    index = {}
    if not folder.exists():
        return index
    for path in folder.iterdir():
        if not path.is_file():
            continue
        stem = path.stem  # DATA_100 or MASK_100
        if not stem.upper().startswith(prefix.upper() + "_"):
            continue
        tail = stem[len(prefix) + 1 :]
        if tail.isdigit():
            index[int(tail)] = path
    return index


def find_case_paths(
    data_dir: Path,
    mask_dir: Path,
    case_id: str,
    data_index: dict = None,
    mask_index: dict = None,
):
    exts = (".nrrd", ".nii", ".nii.gz")
    for cid in case_id_variants(case_id):
        for ext in exts:
            img_path = data_dir / f"DATA_{cid}{ext}"
            mask_path = mask_dir / f"MASK_{cid}{ext}"
            if img_path.exists() and mask_path.exists():
                return img_path, mask_path

    # 回退：按数字 id 在目录索引里查找（兼容 DATA_000.nrrd 等）
    if case_id.isdigit() and data_index and mask_index:
        num = int(case_id)
        if num in data_index and num in mask_index:
            return data_index[num], mask_index[num]

    return None, None


def resolve_paths_from_row(row, split_root: Path, data_dir: Path, mask_dir: Path, case_id: str, data_index, mask_index):
    """优先使用 labels.csv 里的 data_path / mask_path（官方 TDSC-ABUS 格式）。"""
    for col_img, col_mask in (("data_path", "mask_path"), ("image_path", "mask_path")):
        if col_img not in row.index or col_mask not in row.index:
            continue
        raw_img = row[col_img]
        raw_mask = row[col_mask]
        if pd.isna(raw_img) or pd.isna(raw_mask):
            continue
        # CSV 可能是 Windows 反斜杠：DATA\DATA_000.nrrd
        img_rel = Path(str(raw_img).replace("\\", "/"))
        mask_rel = Path(str(raw_mask).replace("\\", "/"))
        img_path = (split_root / img_rel).resolve()
        mask_path = (split_root / mask_rel).resolve()
        if img_path.exists() and mask_path.exists():
            return img_path, mask_path

    return find_case_paths(data_dir, mask_dir, case_id, data_index=data_index, mask_index=mask_index)


def stack_tumor_slices(img_path: Path, mask_path: Path):
    img_sitk = sitk.ReadImage(str(img_path))
    mask_sitk = sitk.ReadImage(str(mask_path))

    img_vol = sitk.GetArrayFromImage(img_sitk)
    mask_vol = sitk.GetArrayFromImage(mask_sitk)

    if img_vol.shape != mask_vol.shape:
        raise ValueError(
            f"Shape mismatch for {img_path.name}: image {img_vol.shape} vs mask {mask_vol.shape}"
        )

    tumor_indices = np.where(mask_vol.reshape(mask_vol.shape[0], -1).sum(axis=1) > 0)[0]
    if len(tumor_indices) == 0:
        return None, 0

    stacked = img_vol[tumor_indices]
    out_img = sitk.GetImageFromArray(stacked.astype(np.float32))

    spacing = list(img_sitk.GetSpacing())
    origin = list(img_sitk.GetOrigin())
    direction = list(img_sitk.GetDirection())

    if len(spacing) == 3:
        out_img.SetSpacing((spacing[0], spacing[1], spacing[2]))
        out_img.SetOrigin(origin)
        out_img.SetDirection(direction)

    return out_img, int(len(tumor_indices))


def process_split(
    split_key: str,
    split_dir_name: str,
    data_root: Path,
    out_img_dir: Path,
    labels_dict: dict,
    split_keys: dict,
    skip_existing: bool,
):
    split_root = data_root / split_dir_name
    data_dir = split_root / "DATA"
    mask_dir = split_root / "MASK"
    csv_path = split_root / "labels.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing labels.csv: {csv_path}")

    df = pd.read_csv(csv_path)
    keys = []
    data_index = build_file_index(data_dir, "DATA")
    mask_index = build_file_index(mask_dir, "MASK")

    if len(data_index) == 0:
        logging.error(
            "%s/DATA 下未找到 DATA_*.nrrd 文件。请确认 Train 数据是否已上传到服务器。",
            split_root,
        )
        if data_dir.exists():
            sample = sorted(p.name for p in data_dir.iterdir())[:5]
            logging.error("DATA 目录现有文件示例: %s", sample)
        else:
            logging.error("目录不存在: %s", data_dir)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split_key}"):
        case_id = normalize_case_id(row["case_id"])
        if not case_id:
            logging.warning("Skip empty case_id in %s", csv_path)
            continue
        case_name = f"ABUS_{case_id}"
        out_path = out_img_dir / f"{case_name}_0000.nii.gz"

        if skip_existing and out_path.exists():
            labels_dict[case_name] = parse_label(row["label"])
            keys.append(case_name)
            continue

        img_path, mask_path = resolve_paths_from_row(
            row, split_root, data_dir, mask_dir, case_id, data_index, mask_index
        )
        if img_path is None:
            logging.warning(
                "Skip %s: image/mask not found in %s (tried DATA_/MASK_ variants for id=%s)",
                case_id,
                split_root,
                case_id,
            )
            continue

        try:
            out_img, num_slices = stack_tumor_slices(img_path, mask_path)
        except Exception as exc:
            logging.warning("Skip %s: %s", case_id, exc)
            continue

        if out_img is None:
            logging.warning("Skip %s: no tumor-containing slices", case_id)
            continue

        sitk.WriteImage(out_img, str(out_path), useCompression=True)
        labels_dict[case_name] = parse_label(row["label"])
        keys.append(case_name)
        logging.info("Saved %s (%d tumor slices)", out_path.name, num_slices)

    split_keys[split_key] = keys
    logging.info("%s: %d cases exported", split_key, len(keys))


def main():
    parser = argparse.ArgumentParser(description="ABUS NRRD -> tumor-slice NIfTI for MST")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/huyiding/pengdie/abus",
        help="ABUS dataset root (contains Train/Validation/Test)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Output root (default: {data_root}/preprocessed_mst)",
    )
    parser.add_argument("--skip_existing", action="store_true", help="Skip cases already converted")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_root = Path(args.data_root)
    output_root = Path(args.output_root or (data_root / "preprocessed_mst"))
    out_img_dir = output_root / "imagesTr"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    labels_dict = {}
    split_keys = {"train": [], "val": [], "test": []}

    for split_key, split_dir_name in SPLIT_DIRS.items():
        process_split(
            split_key=split_key,
            split_dir_name=split_dir_name,
            data_root=data_root,
            out_img_dir=out_img_dir,
            labels_dict=labels_dict,
            split_keys=split_keys,
            skip_existing=args.skip_existing,
        )

    with open(output_root / "classification_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels_dict, f, indent=2, ensure_ascii=False)

    with open(output_root / "split.json", "w", encoding="utf-8") as f:
        json.dump(split_keys, f, indent=2, ensure_ascii=False)

    logging.info("Done. Output: %s", output_root)
    logging.info(
        "Counts -> train: %d, val: %d, test: %d",
        len(split_keys["train"]),
        len(split_keys["val"]),
        len(split_keys["test"]),
    )


if __name__ == "__main__":
    main()
