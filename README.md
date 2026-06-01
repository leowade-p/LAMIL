# LA-MIL: Lesion-Aware Multi-Instance Learning

Code for 3D breast tumor benign–malignant classification on **CBBCT** (5-fold cross-validation) and **ABUS** (fixed train/val/test split), with DINOv2+LoRA segmentation and multiple MIL baselines.

## Repository layout

```
├── shared/              # DINOv2+LoRA, MIL models, RRT-MIL vendor code
├── cbbct/               # CBBCT 5-fold pipeline (DINOv2+LoRA segmentation)
│   ├── configs/         # config.py, config_abmil.py, ...
│   ├── dataloader/      # preprocessing
│   ├── utils/           # logging, folds, metrics helpers
│   ├── pipelines/       # segmentation + classification
│   ├── methods/         # (MIL backends live in shared/)
│   ├── scripts/         # main_*.py, run_*_experiment.py
│   ├── gt/              # ablation: MIL with ground-truth masks (no DINO seg)
│   │   ├── configs/ dataloader/ utils/ pipelines/ methods/ scripts/
│   └── baselines/
│       └── resnet3d/    # 3D R3D-18 end-to-end baseline
├── abus/                # ABUS fixed-split pipeline
│   ├── configs/
│   ├── dataloader/      # slice CSV + PNG loading
│   ├── utils/
│   ├── pipelines/
│   ├── methods/         # mil_model_factory, mil_eval
│   ├── scripts/
│   └── mst/             # optional 3D MST baseline
└── third_party/         # dinov2 clone + pretrained .pth (not in git)
```

Top-level files such as `cbbct/main_component.py` and `abus/config.py` are thin **compatibility shims** that re-export from the subpackages above, so existing commands still work.

## Setup

1. Create a Python 3.10+ environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Clone DINOv2 locally (required for `source="local"` hub loading):

   ```bash
   git clone https://github.com/facebookresearch/dinov2 third_party/dinov2
   ```

3. Download `dinov2_vitl14_reg4_pretrain.pth` into `third_party/` (or set `DINOV2_WEIGHTS`).
4. The datasets can be accessed via Baidu Netdisk:

Link: https://pan.baidu.com/s/1ruwW21I1JRlDkQCdJ8BMmg

Extraction code: Please contact Yineng Zheng for the extraction code.

Contact person: Yineng Zheng
Email: yinengzheng@cqmu.edu.cn
Institution: The First Affiliated Hospital of Chongqing Medical University

5. Configure paths via environment variables (see `cbbct/configs/config.example.py` and `abus/configs/config.example.py`), or edit `cbbct/configs/config.py` / `abus/configs/config.py` (also exposed as `cbbct/config.py`).

| Variable | Description |
|----------|-------------|
| `DINOV2_REPO` | Path to cloned `dinov2` repo |
| `DINOV2_WEIGHTS` | Path to `dinov2_vitl14_reg4_pretrain.pth` |
| `CBBCT_DATA_ROOT` | CBBCT detection/PNG dataset root |
| `CBBCT_WORKING_DIR` | CBBCT outputs and fold checkpoints |
| `ABUS_DATA_ROOT` | ABUS dataset root (`Train/`, `Validation/`, `Test/`) |

## CBBCT (5-fold)

From the repository root:

```bash
# Main LA-MIL experiment (requires per-fold segmentation checkpoints)
python cbbct/main_component.py --roi 0.0

# Train segmentation inside each fold (optional)
python cbbct/main_kfold.py

# MIL baselines
python cbbct/run_abmil_experiment.py --roi 0.0
```

Per-fold segmentation weights must match `SEG_MODEL_PATH_TEMPLATE` in `cbbct/config.py`.

### CBBCT GT-mask ablation (`cbbct/gt/`)

Uses panoptic **ground-truth masks** instead of predicted segmentation (paper Tables ablation). No DINOv2 segmentation step.

```bash
python cbbct/gt/main_kfold.py
```

Configure `CBBCT_DATA_ROOT`, `CBBCT_GT_WORKING_DIR`, and `MASK_DIRS` in `cbbct/gt/configs/config.py`.

### CBBCT 3D ResNet baseline (`cbbct/baselines/resnet3d/`)

End-to-end **R3D-18** on 3D volumes (comparison method in the paper).

```bash
python cbbct/baselines/resnet3d/train.py
```

Set `CBBCT_RESNET3D_DATA` to your `Dataset510_TumorTotal` (or equivalent) directory with `imagesTr/` and `pid_map_total.json`.

## ABUS (fixed split)

```bash
cd abus
python main_component.py
# or run all MIL methods:
python run_all_mil_experiments.py --roi 0.0
```

Expected layout under `ABUS_DATA_ROOT`:

- `Train/labels.csv`, `Validation/labels.csv`, `Test/labels.csv`
- `preprocessed_png_data_1.30/{train,val,test}/{images,masks}/`

## MST baseline (ABUS, optional)

```bash
cd abus/mst/scripts
# Edit dataset paths in main_train_abus.py / dataset_3d_abus.py first
python main_train_abus.py
```

Install optional dependencies listed in `requirements.txt` (commented section).

## Notes

- Do **not** commit patient imaging data or trained checkpoints.
- `cbbct/config.py` includes `ALL_PATIENTS_LABELS` for reproducible 5-fold splits (IDs only).
- Third-party full repos (CLAM, TransMIL, etc.) are not required; MIL implementations live under `shared/`.
