# utils.py
import logging
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch

import config


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Global random seed set to {seed}")


def setup_logging(fold_identifier: str) -> None:
    log_dir = os.path.join(config.OUTPUT_DIR, str(fold_identifier))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info(f"Logging configured. Log file: {log_file}")


def summarize_results(results: Union[Dict[str, Any], List[Dict[str, Any]]]):
    if not results:
        logging.info("No results to summarize.")
        return

    if isinstance(results, dict):
        results = [results]

    results_df = pd.DataFrame(results)
    logging.info("\n" + "=" * 50)
    logging.info(" Final Results Summary ".center(50, "="))
    logging.info("=" * 50)
    logging.info("\n" + results_df.round(4).to_string(index=False))

    csv_path = os.path.join(config.OUTPUT_DIR, "final_results_summary.csv")
    results_df.to_csv(csv_path, index=False)
    logging.info(f"Results saved to {csv_path}")


def get_segmentation_model_path() -> str:
    if config.SEGMENTATION_MODEL_PATH.strip():
        return config.SEGMENTATION_MODEL_PATH.strip()
    return os.path.join(config.OUTPUT_DIR, str(config.FOLD_IDENTIFIER), "best_segmentation_model.pt")
