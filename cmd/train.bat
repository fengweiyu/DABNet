@echo off
setlocal enabledelayedexpansion
cd ..
:: 保存当前目录
set "CURRENT_DIR=%cd%"

:: 进入目标目录获取文件
cd /d "checkpoint\cityscapes\DABNetbs8gpu1_train"
set "LATEST_PTH="
for /f "delims=" %%a in ('dir /b /o-d *.pth 2^>nul') do (
    set "LATEST_PTH=%%a"
    goto :got
)
:got

:: 返回原目录
cd /d "%CURRENT_DIR%"

:: 使用获取到的文件名
if defined LATEST_PTH (
    echo 在子目录中找到: !LATEST_PTH!
    
    :: 激活conda环境
    call conda activate torchCV
    
    :: 运行训练脚本（注意使用 !LATEST_PTH! 而不是 %LATEST_PTH%）
    python train.py --dataset cityscapes --max_epochs 1000 --resume "./checkpoint/cityscapes/DABNetbs8gpu1_train/!LATEST_PTH!"
) else (
    echo 未找到.pth文件，将从头开始训练
    call conda activate torchCV
    python train.py --dataset cityscapes --max_epochs 1000
)

pause