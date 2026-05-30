# main_fixed_seeds.py

import scripts._bootstrap  # noqa: F401

import os
import logging
import pandas as pd
import glob
import numpy as np
import random
import torch
import config
import utils
import segmentation_trainer
import classification_pipeline

# ==============================================================================
# 1. 新的数据加载函数 (保持不变)
# ==============================================================================
def load_slice_data(split_name: str, png_data_root: str, csv_path: str) -> list:
    # ... (这部分代码与你提供的一模一样，为了节省篇幅略去) ...
    logging.info(f"Loading 2D slice data for '{split_name}' split...")
    
    if not os.path.exists(csv_path):
        logging.error(f"Label CSV file not found at: {csv_path}")
        return []
        
    df = pd.read_csv(csv_path)
    label_map = {
        str(row['case_id']): 1 if str(row['label']).upper() in ['M', '1'] else 0 
        for _, row in df.iterrows()
    }
    
    image_dir = os.path.join(png_data_root, split_name.lower(), 'images')
    mask_dir = os.path.join(png_data_root, split_name.lower(), 'masks')
    
    if not os.path.exists(image_dir):
        logging.error(f"Image directory not found at: {image_dir}")
        return []

    slice_list = []
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        try:
            parts = filename.split('_')
            pid = parts[1]
            mask_filename = filename.replace('.png', '_mask.png')
            mask_path = os.path.join(mask_dir, mask_filename)

            if not os.path.exists(mask_path): continue

            patient_label = label_map.get(pid)
            if patient_label is None: continue
            
            slice_list.append({
                'image_path': img_path,
                'mask_path': mask_path,
                'pid': pid,
                'label': patient_label
            })
        except IndexError:
            continue
            
    logging.info(f"Found {len(slice_list)} slices for '{split_name}' split.")
    return slice_list

# ==============================================================================
# 2. 主流程函数 (修改版：支持多种子循环)
# ==============================================================================
def main():
    # 1. 基础设置
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    utils.setup_logging("abus_multi_seed_roi")
    logging.info("#################### STARTING ABUS PIPELINE (MULTI-SEED) ####################")

    # 2. 加载数据 (只加载一次，不用在循环里重复加载)
    logging.info("Loading 2D slice datasets...")
    train_slice_list = load_slice_data("train", config.PREPROCESSED_DATA_ROOT, config.TRAIN_CSV)
    val_slice_list   = load_slice_data("val", config.PREPROCESSED_DATA_ROOT, config.VAL_CSV)
    test_slice_list  = load_slice_data("test", config.PREPROCESSED_DATA_ROOT, config.TEST_CSV)

    if not train_slice_list:
        logging.error("Training data list is empty.")
        return

    # 3. 定义30个随机种子
    # 固定种子列表以确保可复现性，或者随机生成
    # seeds = [random.randint(1, 10000) for _ in range(30)]
    # rois = [
    #         0.0001, 0.0002, 0.0005,
    # 0.001, 0.002, 0.005,
    # 0.01, 0.02, 0.05,
    # 0.1, 0.2,0.5
    # ]
    # rois = [0.0002, 0.0004, 0.00005,0.00008,0.00007
    # ]
    rois = [0.0001]
    # roi  = [
    #     43,
    #     # 41,42,43,99
    #     # 36,35,37,34,38,39,40,46,47,48,49,50,51,52,
    # ]
    
    # 结果容器
    all_results = []
    
    # 假设分割模型是固定的（只训练分类器稳定性），指定分割模型路径
    # 如果分割模型也要重新训练，请取消下面注释并放入循环
    fixed_segmentation_path = "/home/huyiding/pengdie/abus/abus_fixed_split_results/abus_run/best_segmentation_model.pt"
    
    if not os.path.exists(fixed_segmentation_path):
        logging.warning(f"Segmentation model not found at {fixed_segmentation_path}, make sure to train it first or update path.")

    # 4. 开始循环训练
    for i, roi in enumerate(rois):
        seed=43
        fold_identifier = f"abus_run_seed_{seed}"
        logging.info(f"\n{'='*20} Running Seed {roi} ({i+1}/{len(rois)}) {'='*20}")
        
        # [关键] 设置随机种子
        utils.set_seed(seed)
        
        # --- 步骤 A: 分割模型 ---
        # 如果你想保持分割模型不变（推荐，为了解耦），直接使用上面的路径
        segmentation_model_path = fixed_segmentation_path
        
        # 如果你想每次都重新训练分割模型（非常耗时），请取消下面的注释：
        # segmentation_model_path = segmentation_trainer.run_segmentation_training(
        #     fold_identifier=fold_identifier,
        #     train_info=train_slice_list,
        #     val_info=val_slice_list,
        #     test_info=test_slice_list
        # )

        # --- 步骤 B: 训练分类器 ---
        # 注意：这里会重新提取特征（因为包含随机增强），所以每次特征都不一样
        try:
            logging.info(f"--- Starting Classification Training for Seed {seed} ---")
            classifier_model_path = classification_pipeline.run_classification_training(
                fold_identifier=fold_identifier,
                train_info=train_slice_list,
                val_info=val_slice_list,
                segmentation_model_path=segmentation_model_path,
                roi=roi
            )
            
            # --- 步骤 C: 测试 ---
            logging.info(f"--- Starting Testing for Seed {seed} ---")
            test_metrics = classification_pipeline.run_classification_testing(
                fold_identifier=fold_identifier,
                test_info=test_slice_list,
                segmentation_model_path=segmentation_model_path,
                classifier_model_path=classifier_model_path
            )
            
            # 记录结果
            result_entry = test_metrics.copy()
            result_entry['seed'] = seed
            all_results.append(result_entry)
            
            logging.info(f"Seed {seed} result: AUC={test_metrics.get('auc', 0):.4f}")
            
        except Exception as e:
            logging.error(f"Error occurred during run with seed {seed}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 5. 汇总统计
    logging.info("\n#################### FINAL SUMMARY ####################")
    if len(all_results) > 0:
        df_results = pd.DataFrame(all_results)
        
        # 计算均值和标准差
        summary = df_results.describe().transpose()[['mean', 'std', 'min', 'max']]
        logging.info("Metrics Statistics over 30 runs:")
        logging.info(f"\n{summary}")
        
        # 保存详细结果到 CSV
        summary_csv_path = os.path.join(config.OUTPUT_DIR, "multi_seed_results_summary.csv")
        df_results.to_csv(summary_csv_path, index=False)
        logging.info(f"Detailed results saved to {summary_csv_path}")
        
        # 打印关键指标
        mean_auc = df_results['auc'].mean()
        std_auc = df_results['auc'].std()
        print(f"\nFinal Result (30 runs): AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    else:
        logging.error("No results collected.")
    logging.info(config.CLS_EPOCHS)

if __name__ == "__main__":
    main()