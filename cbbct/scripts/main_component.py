# main_component.py

import scripts._bootstrap  # noqa: F401

import os
import logging
from typing import List
import argparse
# 导入我们创建的所有模块
import config
import utils
import new_classification_pipeline
def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='基于3D连通分量的5折交叉验证流程')
    
    # 添加景区参数
    parser.add_argument('--roi',  type=float, required=True,
                       help='roi weight')
    # parser.add_argument('--lr',  type=float, required=True,
    #                    help=' ')
    
    return parser.parse_args()
def main(args=None):
    """
    主函数，用于执行基于3D连通分量的5折交叉验证流程。
    本流程假设分割模型已为每一折预先训练好。
    """
    # 1. 初始化设置
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    utils.set_seed(config.RANDOM_SEED)
    
    # 2. 在病人层面划分数据集为5折
    patient_folds: List[List[str]] = utils.create_patient_folds(
        patient_labels=config.ALL_PATIENTS_LABELS,
        num_folds=config.K_FOLDS,
        seed=config.RANDOM_SEED
    )

    all_fold_results = []
    roi = args.roi

    # 3. 主循环，迭代5次
    for i in range(config.K_FOLDS):
        fold_num = i + 1
        utils.setup_logging(fold_num)
        logging.info(f"#################### COMPONENT-BASED: FOLD {fold_num}/{config.K_FOLDS} ####################")

        # 4. 数据集角色分配
        test_fold_index = i
        val_fold_index = (i + 1) % config.K_FOLDS
        train_fold_indices = [j for j in range(config.K_FOLDS) if j != test_fold_index and j != val_fold_index]

        test_pids = patient_folds[test_fold_index]
        val_pids = patient_folds[val_fold_index]
        train_pids = [pid for idx in train_fold_indices for pid in patient_folds[idx]]

        logging.info(f"Data Split for Fold {fold_num}:")
        logging.info(f"  - Test Set     : Fold {test_fold_index + 1} ({len(test_pids)} patients)")
        logging.info(f"  - Validation Set : Fold {val_fold_index + 1} ({len(val_pids)} patients)")
        logging.info(f"  - Training Set   : Folds {[idx + 1 for idx in train_fold_indices]} ({len(train_pids)} patients)")

        # --- 执行新的流水线 ---

        # 步骤 A: 定位并检查预训练的分割模型
        segmentation_model_path = config.SEG_MODEL_PATH_TEMPLATE.format(fold_num=fold_num)
        logging.info(f"Attempting to load pre-trained segmentation model for this fold from: {segmentation_model_path}")
        if not os.path.exists(segmentation_model_path):
            logging.error(f"FATAL: Segmentation model not found at the specified path for Fold {fold_num}. Please ensure it is pre-trained and saved correctly.")
            logging.error("Skipping this fold.")
            continue
        logging.info("Segmentation model found.")

        # 步骤 B: 训练分类器
        # 这个函数现在会加载分割模型，提取基于连通分量的特征，并训练新的PatientLevelClassifier
        classifier_model_path = new_classification_pipeline.run_classification_training(
            fold_num=fold_num,
            train_pids=train_pids,
            val_pids=val_pids,
            segmentation_model_path=segmentation_model_path,
            lr=config.CLS_LR,
            roi = roi,
 
        )
        # fold_dir = os.path.join(config.OUTPUT_DIR, f"fold_{fold_num}")
        # best_model_path = os.path.join(fold_dir, "best_mil+_classifier_model.pth")
        # 步骤 C: 在测试集上评估
        # 这个函数会加载最好的分类器，在独立的测试集上进行最终评估
        test_metrics = new_classification_pipeline.run_classification_testing(
            fold_num=fold_num,
            test_pids=test_pids,
            segmentation_model_path=segmentation_model_path,
            classifier_model_path=classifier_model_path
        )
        
        all_fold_results.append(test_metrics)
        logging.info(f"#################### FOLD {fold_num} COMPLETED ####################\n")

    # 5. 总结所有折的结果
    logging.info("All k-fold iterations have completed. Summarizing results.")
    utils.summarize_results(all_fold_results)
    logging.info(config.MAX_ROIS)
    logging.info(config.CLS_HIDDEN_DIM)
    logging.info(config.CLS_N_HEADS)
    logging.info(config.CLS_LR)
    logging.info('layers  数')
    logging.info(config.CLS_N_LAYERS)



    
    logging.info(config.CLS_EPOCHS)
    


if __name__ == "__main__":
    args = parse_arguments()
    main(args=args)

# main_component.py

# import os
# import logging
# from typing import List

# # 导入我们创建的所有模块
# import config
# import utils
# import new_classification_pipeline

# def main():
#     """
#     主函数，用于执行基于3D连通分量的5折交叉验证流程。
#     本流程假设分割模型已为每一折预先训练好。
#     """
#     # 1. 初始化设置
#     os.makedirs(config.OUTPUT_DIR, exist_ok=True)
#     utils.set_seed(config.RANDOM_SEED)

#     # 2. 在病人层面划分数据集为5折
#     patient_folds: List[List[str]] = utils.create_patient_folds(
#         patient_labels=config.ALL_PATIENTS_LABELS,
#         num_folds=config.K_FOLDS,
#         seed=config.RANDOM_SEED
#     )

#     all_fold_results = []

#     # 3. 主循环，迭代5次
#     for i in range(config.K_FOLDS):
#         fold_num = i + 1
#         utils.setup_logging(fold_num)
#         logging.info(f"#################### COMPONENT-BASED: FOLD {fold_num}/{config.K_FOLDS} ####################")

#         # 4. 数据集角色分配
#         test_fold_index = i
#         val_fold_index = (i + 1) % config.K_FOLDS
#         train_fold_indices = [j for j in range(config.K_FOLDS) if j != test_fold_index and j != val_fold_index]

#         test_pids = patient_folds[test_fold_index]
#         val_pids = patient_folds[val_fold_index]
#         train_pids = [pid for idx in train_fold_indices for pid in patient_folds[idx]]

#         logging.info(f"Data Split for Fold {fold_num}:")
#         logging.info(f"  - Test Set     : Fold {test_fold_index + 1} ({len(test_pids)} patients)")
#         logging.info(f"  - Validation Set : Fold {val_fold_index + 1} ({len(val_pids)} patients)")
#         logging.info(f"  - Training Set   : Folds {[idx + 1 for idx in train_fold_indices]} ({len(train_pids)} patients)")

#         # --- 执行新的流水线 ---

#         # 步骤 A: 定位并检查预训练的分割模型
#         segmentation_model_path = config.SEG_MODEL_PATH_TEMPLATE.format(fold_num=fold_num)
#         logging.info(f"Attempting to load pre-trained segmentation model for this fold from: {segmentation_model_path}")
#         if not os.path.exists(segmentation_model_path):
#             logging.error(f"FATAL: Segmentation model not found at the specified path for Fold {fold_num}. Please ensure it is pre-trained and saved correctly.")
#             logging.error("Skipping this fold.")
#             continue
#         logging.info("Segmentation model found.")

#         # 步骤 B: 训练分类器
#         # 这个函数现在会加载分割模型，提取基于连通分量的特征，并训练新的PatientLevelClassifier
#         classifier_model_path = new_classification_pipeline.run_classification_training(
#             fold_num=fold_num,
#             train_pids=train_pids,
#             val_pids=val_pids,
#             segmentation_model_path=segmentation_model_path
#         )
        
#         # 步骤 C: 在测试集上评估
#         # 这个函数会加载最好的分类器，在独立的测试集上进行最终评估
#         test_metrics = new_classification_pipeline.run_classification_testing(
#             fold_num=fold_num,
#             test_pids=test_pids,
#             segmentation_model_path=segmentation_model_path,
#             classifier_model_path=classifier_model_path
#         )
        
#         all_fold_results.append(test_metrics)
#         logging.info(f"#################### FOLD {fold_num} COMPLETED ####################\n")

#     # 5. 总结所有折的结果
#     logging.info("All k-fold iterations have completed. Summarizing results.")
#     utils.summarize_results(all_fold_results)
#     logging.info(config.MAX_ROIS)

# if __name__ == "__main__":
#     main()