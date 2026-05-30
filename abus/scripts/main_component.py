# main_component.py - ABUS 固定划分 MIL 实验入口
import scripts._bootstrap  # noqa: F401

import argparse
import logging
import os

import classification_pipeline
import config
import data_loader
import segmentation_trainer
import utils


def parse_arguments():
    parser = argparse.ArgumentParser(description="ABUS 2D 工作流 MIL 实验")
    parser.add_argument("--roi", type=float, default=0.0, help="ROI loss 权重")
    parser.add_argument("--w-plus", type=float, default=None, help="不对称 MIL 正样本权重 (仅 mil_asym)")
    parser.add_argument(
        "--train-seg",
        action="store_true",
        help="强制重新训练分割模型（默认使用已有权重）",
    )
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_arguments()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    utils.set_seed(config.RANDOM_SEED)
    utils.setup_logging(config.FOLD_IDENTIFIER)

    logging.info("#################### ABUS MIL EXPERIMENT START ####################")
    logging.info(f"Method: {config.CLS_MODEL_TYPE}")
    logging.info(f"Output: {config.OUTPUT_DIR}")

    train_slice_list = data_loader.load_slice_data("train", config.PREPROCESSED_DATA_ROOT, config.TRAIN_CSV)
    val_slice_list = data_loader.load_slice_data("val", config.PREPROCESSED_DATA_ROOT, config.VAL_CSV)
    test_slice_list = data_loader.load_slice_data("test", config.PREPROCESSED_DATA_ROOT, config.TEST_CSV)

    logging.info(f"Train slices: {len(train_slice_list)} | Val: {len(val_slice_list)} | Test: {len(test_slice_list)}")
    if not train_slice_list:
        logging.error("Training data is empty. Check PREPROCESSED_DATA_ROOT and CSV paths in config.py")
        return None

    seg_path = utils.get_segmentation_model_path()
    if args.train_seg or config.TRAIN_SEGMENTATION or not os.path.exists(seg_path):
        logging.info("--- Training segmentation model ---")
        seg_path = segmentation_trainer.run_segmentation_training(
            fold_identifier=config.FOLD_IDENTIFIER,
            train_info=train_slice_list,
            val_info=val_slice_list,
            test_info=test_slice_list,
        )
    else:
        logging.info(f"--- Using existing segmentation model: {seg_path} ---")

    logging.info("--- Training MIL classifier ---")
    classifier_path = classification_pipeline.run_classification_training(
        fold_identifier=config.FOLD_IDENTIFIER,
        train_info=train_slice_list,
        val_info=val_slice_list,
        segmentation_model_path=seg_path,
        roi=args.roi,
        w_plus=args.w_plus,
    )

    logging.info("--- Testing MIL classifier ---")
    test_metrics = classification_pipeline.run_classification_testing(
        fold_identifier=config.FOLD_IDENTIFIER,
        test_info=test_slice_list,
        segmentation_model_path=seg_path,
        classifier_model_path=classifier_path,
    )

    utils.summarize_results(test_metrics)
    logging.info("#################### ABUS MIL EXPERIMENT DONE ####################")
    return test_metrics


if __name__ == "__main__":
    main()
