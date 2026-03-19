# train.py
# 修改理由：将单次训练改为“循环跑四个yaml消融实验”，并让每个实验输出目录=yaml文件名，方便对比与管理。
# 修改理由：固定随机种子 + PyTorch确定性设置，尽可能保证可复现（A800上也适用，代价是速度略降）。
# 修改理由：按A800(80GB) + 640分割 + 4000张数据的规模，设置更合适的batch、学习率与训练超参。

import os
import random
from pathlib import Path

import numpy as np
import torch

from ultralytics import YOLO


def set_seed(seed: int = 42, deterministic: bool = True):
    """固定随机种子与确定性设置（尽可能复现）."""
    # 修改理由：固定python/np/torch随机性，减少实验波动，保证消融对比可信。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 修改理由：控制cuDNN与cuBLAS的确定性行为（注意：会降低一些速度，但提高可复现性）。
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # 某些算子不支持完全确定性时，仍尽最大可能固定
            pass


def main():
    # =========================
    # 全局配置（你按需改路径即可）
    # =========================
    data_yaml = "/public/home/ctjby/lxyolov8-0301/ultralytics-main/ultralytics/cfg/datasets/crack_seg_fly.yaml"
    pretrained = "yolov8n-seg.pt"  # 修改理由：保持与原训练一致，确保对比公平。
    device = 0  # 修改理由：明确使用一张A800（cuda:0）。

    # 你的四个消融实验yaml（文件名会自动用于runs输出目录）
    exp_yamls = [
        "yolov8-seg.yaml",
        "yolov8-seg-spd13-mdka-sadilation-sppf-replk-full.yaml",
        "yolov8-seg-spd13.yaml",
        "yolov8-seg-sppf-replk-full.yaml",
        "yolov8-seg-mdka-sadilation.yaml",
        # "yolov8-seg-mdka-neck.yaml",       # 改进1：只替换输出三尺度C2f为C2fMDKA
        # "yolov8-seg-mdka-gated.yaml",      # 改进2：门控融合 gated
        # "yolov8-seg-mdka-boundary.yaml",   # 改进3-B1：边界增强 boundary
        # "yolov8-seg-mdka-sadilation.yaml",
        # yolov8-seg-mcsta-p2.yaml，yolov8-seg-agm.yaml，yolov8-seg-ciepool-afp-secbam.yaml,yolov8-seg-slpa-msfem.yaml # 改进3-B2：尺度自适应 dilations
    ]

    # =========================
    # 复现性设置
    # =========================
    seed = 42
    set_seed(seed=seed, deterministic=True)

    # =========================
    # 训练超参建议（A800 + 640 seg）
    # =========================
    imgsz = 640

    # 修改理由：A800显存充足；yolov8n-seg在640分割通常可用更大batch提升稳定性与效率。
    # 经验推荐：batch=64 作为起点（如果你模块更重或显存占用更大，降到32）。
    batch = 16

    # 修改理由：Ultralytics默认lr0对batch=64附近较合适；消融对比中保持相同lr更公平。
    # 若你将batch改为32，可把lr0改为0.005；若batch改为16，lr0改为0.0025（线性缩放）。
    lr0 = 0.0025

    # 修改理由：4000张数据，200 epochs 通常足够收敛并观察差异；你原来就是200，保持可比。
    epochs = 200

    # 修改理由：A800上建议开启AMP提升速度与吞吐；若你遇到不稳定/NaN再改回False。
    amp = True

    # 修改理由：分割任务通常更依赖稳定训练；patience适中即可。
    patience = 50

    # 修改理由：workers按服务器CPU情况调整。一般8~16；过高可能引发IO抖动。
    workers = 8

    # =========================
    # 依次跑四个实验
    # =========================
    for yaml_path in exp_yamls:
        yaml_stem = Path(yaml_path).stem

        # 修改理由：每个实验重新构建模型，避免权重/缓存串扰。
        model = YOLO(yaml_path).load(pretrained)

        # 修改理由：输出目录按yaml命名，便于对比：runs/segment/<yaml_stem>/
        # Ultralytics默认project=runs/segment，这里显式指定，保证路径稳定。
        model.train(
            data=data_yaml,
            imgsz=imgsz,
            device=device,
            seed=seed,
            deterministic=True,  # 修改理由：Ultralytics内部也提供确定性选项，双保险。
            amp=amp,
            batch=batch,
            lr0=lr0,
            epochs=epochs,
            patience=patience,
            workers=workers,
            # 修改理由：保存与验证开启，方便选择best并对比曲线。
            save=True,
            val=True,
            # 修改理由：每个实验的runs目录独立，名称直接用yaml名避免混淆。
            project="runs/segment",
            name=yaml_stem,
            exist_ok=False,  # 修改理由：防止覆盖旧实验；若你想覆盖改为True。
        )

        # 可选：训练完释放显存，避免长循环显存碎片化
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
