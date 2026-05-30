#!/bin/bash

# 1. 设置 WandB 为离线模式，防止弹窗卡住脚本
export WANDB_MODE=offline

# 定义一些公共参数 (根据你的 parser 默认值，这里可以显式指定以防万一)
DATASET="TUMOR"
MODEL="DinoV2ClassifierSlice"
OUTPUT_DIR="./runs"

echo "========================================"
echo "开始 5 折交叉验证训练 (Fold 0 - 4)"
echo "模型: $MODEL | 数据集: $DATASET"
echo "========================================"

# 2. 循环 0 到 4
for fold in {4..4}
do
    echo ""
    echo "----------------------------------------"
    echo "正在启动 Fold $fold ..."
    echo "----------------------------------------"
    
    # 运行 Python 命令
    # 注意：这里使用 -m scripts.main_train_kfold
    python -m scripts.main_train_kfold \
        --dataset $DATASET \
        --model $MODEL \
        --path_root_output $OUTPUT_DIR \
        --fold $fold
    
    # 检查上一条命令是否执行成功
    if [ $? -eq 0 ]; then
        echo "Fold $fold 训练完成。"
    else
        echo "Fold $fold 训练出错！停止脚本。"
        exit 1
    fi
done

echo ""
echo "========================================"
echo "所有 Fold 训练结束！"
echo "========================================"