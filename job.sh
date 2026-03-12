#!/bin/bash
#SBATCH -J yolov8n_seg_train  # 任务名
#SBATCH -p gpuA800         # 提交到A800 GPU分区
#SBATCH -N 1                 # 单节点
#SBATCH -n 1                 # 单进程
#SBATCH --gres=gpu:1           # 独占1张A800 GPU
#SBATCH --cpus-per-task=12   # A800绑定24核CPU（匹配算力）
#SBATCH --mem=64GB          # 充足主机内存（A800训练需求）
#SBATCH -o %j_train.log      # 输出日志（任务ID命名）
#SBATCH -e %j_train.err      # 错误日志

# 加载集群环境

module load apps/anaconda3/3-2024.10  # 华工HPC标准Anaconda模块
module load cuda/12.4.1       # 适配A800的CUDA版本


# 激活自定义环境（替换为你的环境名/路径）
source activate base
conda activate /public/home/ctjby/lxyolov8-0301/ultralytics-main/yolov8-enviroment

# 进入train.py所在目录（替换为实际路径）
cd /public/home/ctjby/lxyolov8-0301/ultralytics-main  

# 执行YOLOv8n-seg训练（核心训练参数见下文）
python train.py 

