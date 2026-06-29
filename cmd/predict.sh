#!/bin/bash
export OMP_NUM_THREADS=16

# 初始化 conda（脚本中必须，否则 conda activate 不可用）
eval "$(conda shell.bash hook)"

# 保存当前目录
CURRENT_DIR="$(pwd)"

# 进入 checkpoint 目录查找最新的 .pth 文件
CHECKPOINT_DIR="checkpoint/cityscapes/DABNetbs8gpu1_train"
cd "$CHECKPOINT_DIR" || {
    echo "目录 $CHECKPOINT_DIR 不存在，将从头开始训练"
    exit 0
}

LATEST_PTH=$(ls -t *.pth 2>/dev/null | head -1)

# 返回原目录
cd "$CURRENT_DIR"

# 根据是否找到 .pth 文件决定恢复或重新训练
if [ -n "$LATEST_PTH" ]; then
    echo "在子目录中找到: $LATEST_PTH"

    # 激活 conda 环境
    conda activate torchCV4

    # 运行训练脚本（从 checkpoint 恢复）
    #python train.py --dataset cityscapes --max_epochs 1000 --resume "./$CHECKPOINT_DIR/$LATEST_PTH"
	#python test.py --dataset cityscapes --checkpoint "./$CHECKPOINT_DIR/$LATEST_PTH" --best --save
	python predict.py --dataset cityscapes --checkpoint "./$CHECKPOINT_DIR/$LATEST_PTH" 
else
    echo "未找到 .pth 文件，将从头开始训练"
fi
