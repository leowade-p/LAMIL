# main_fixed.py

import scripts._bootstrap  # noqa: F401

import os
import logging
import pandas as pd
import glob
import config
import utils
import segmentation_trainer
import classification_pipeline

# ==============================================================================
# 1. 新的数据加载函数
# ==============================================================================
def load_slice_data(split_name: str, png_data_root: str, csv_path: str) -> list:
    """
    扫描预处理好的PNG目录，并结合CSV文件中的标签，创建一个以切片为单位的数据列表。

    Args:
        split_name (str): 数据集划分的名称 (例如 'train', 'validation', 'test').
        png_data_root (str): 预处理PNG数据的根目录 (例如 './preprocessed_png_data').
        csv_path (str): 包含病人标签的 labels.csv 文件的路径。

    Returns:
        list: 一个包含字典的列表，每个字典代表一个2D切片的信息。
    """
    logging.info(f"Loading 2D slice data for '{split_name}' split...")
    
    # 读取CSV，创建从 case_id 到 label 的映射
    if not os.path.exists(csv_path):
        logging.error(f"Label CSV file not found at: {csv_path}")
        return []
        
    df = pd.read_csv(csv_path)
    label_map = {
        str(row['case_id']): 1 if str(row['label']).upper() in ['M', '1'] else 0 
        for _, row in df.iterrows()
    }
    
    # 构建图像和掩码目录的路径
    image_dir = os.path.join(png_data_root, split_name.lower(), 'images')
    mask_dir = os.path.join(png_data_root, split_name.lower(), 'masks')
    
    if not os.path.exists(image_dir):
        logging.error(f"Image directory not found at: {image_dir}")
        return []

    # 扫描图像目录并构建信息列表
    slice_list = []
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        
        try:
            # 解析文件名: "Train_0_slice_261.png"
            parts = filename.split('_')
            pid = parts[1]
            
            # 构建对应的掩码路径
            mask_filename = filename.replace('.png', '_mask.png')
            mask_path = os.path.join(mask_dir, mask_filename)

            if not os.path.exists(mask_path):
                continue

            patient_label = label_map.get(pid)
            if patient_label is None:
                continue
            
            slice_list.append({
                'image_path': img_path,
                'mask_path': mask_path,
                'pid': pid,
                'label': patient_label
            })
        except IndexError:
            logging.warning(f"Could not parse filename '{filename}', skipping.")
            continue
            
    logging.info(f"Found {len(slice_list)} slices for '{split_name}' split.")
    return slice_list

# ==============================================================================
# 2. 主流程函数 (已更新)
# ==============================================================================
def main():
    # 1. 初始化 (保持不变)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    utils.set_seed(config.RANDOM_SEED)
    utils.setup_logging("abus_fixed")
    logging.info("#################### STARTING ABUS PIPELINE (2D WORKFLOW) ####################")

    # 2. 加载2D切片数据列表
    logging.info("Loading 2D slice datasets...")
    
    # 假设您的 config.py 中有如下路径定义:
    # PREPROCESSED_DATA_ROOT = "./preprocessed_png_data"
    # TRAIN_CSV = "Train/labels.csv"
    # VAL_CSV = "Validation/labels.csv"
    # TEST_CSV = "Test/labels.csv"
    
    train_slice_list = load_slice_data("train", config.PREPROCESSED_DATA_ROOT, config.TRAIN_CSV)
    val_slice_list   = load_slice_data("val", config.PREPROCESSED_DATA_ROOT, config.VAL_CSV)
    test_slice_list  = load_slice_data("test", config.PREPROCESSED_DATA_ROOT, config.TEST_CSV)

    logging.info(f"Data Split Summary (2D Slices):")
    logging.info(f"  - Train : {len(train_slice_list)} slices")
    logging.info(f"  - Val   : {len(val_slice_list)} slices")
    logging.info(f"  - Test  : {len(test_slice_list)} slices")

    # 简单检查文件是否存在
    if not train_slice_list:
        logging.error("Training data list is empty. Please check paths and preprocessed data.")
        return
    if not os.path.exists(train_slice_list[0]['image_path']):
        logging.error(f"Sample file not found: {train_slice_list[0]['image_path']}")
        return

    fold_identifier = "abus_run"

    # 3. 步骤 A: 训练分割模型
    logging.info("--- Starting Segmentation Training ---")
    # segmentation_model_path = segmentation_trainer.run_segmentation_training(
    #     fold_identifier=fold_identifier,
    #     train_info=train_slice_list,  # 传递切片列表
    #     val_info=val_slice_list   ,    # 传递切片列表
    #     test_info = test_slice_list
    # )
    output_dir = os.path.join(config.OUTPUT_DIR, str(fold_identifier))
    os.makedirs(output_dir, exist_ok=True)
    segmentation_model_path = os.path.join(output_dir, "best_segmentation_model.pt")
    # 4. 步骤 B: 训练分类器
    logging.info("--- Starting Classification Training ---")
    classifier_model_path = classification_pipeline.run_classification_training(
        fold_identifier=fold_identifier,
        train_info=train_slice_list,      # 同样使用切片列表
        val_info=val_slice_list,          # 同样使用切片列表
        segmentation_model_path=segmentation_model_path
    )
    classifier_model_path = os.path.join(config.OUTPUT_DIR, str(fold_identifier), "best_mil_classifier.pth")
    # 5. 步骤 C: 测试
    logging.info("--- Starting Testing ---")
    test_metrics = classification_pipeline.run_classification_testing(
        fold_identifier=fold_identifier,
        test_info=test_slice_list,        # 传递测试集的切片列表
        segmentation_model_path=segmentation_model_path,
        classifier_model_path=classifier_model_path
    )
    
    logging.info(f"#################### PIPELINE COMPLETED ####################")
    logging.info(test_metrics)
    logging.info(config.CLS_EPOCHS)

if __name__ == "__main__":
    main()