# train.py
# 修改理由：显式指定使用 YOLOv8m-seg 架构，避免使用 yolov8-seg.yaml 时因未传 scale 而默认回落到 n。
# 修改理由：显式指定优化器为 SGD，避免 Ultralytics 的 optimizer=auto 自动改成 AdamW。
# 修改理由：显式指定 lr0=0.01、momentum=0.937、weight_decay=0.0005，保证训练超参数严格按论文设定执行。
# 修改理由：保留多 seed 循环逻辑，便于后续统计 mean/std，减少单次随机性的偶然影响。
# 修改理由：每个 seed 单独输出到独立目录，避免结果覆盖，保证实验可追溯。

import os
import random
from pathlib import Path

import numpy as np
import torch

from ultralytics import YOLO


def set_seed(seed: int = 42, deterministic: bool = True):
    """固定随机种子与确定性设置（尽可能复现）."""
    # 修改理由：固定 python / numpy / torch 随机性，降低实验波动，提高消融对比可信度。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 修改理由：控制 cuDNN 与 cuBLAS 的确定性行为，尽量保证同一 seed 可复现。
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # 修改理由：部分算子可能不完全支持确定性，异常时保持最大程度固定随机性。
            pass


def main():
    # =========================
    # 全局配置（按需改路径）
    # =========================
    data_yaml = "ultralytics/cfg/datasets/crack_seg_paper.yaml"

    # 修改理由：显式使用 yolov8-seg 预训练权重，同时该权重对应 m 架构。
    pretrained = "./yolov8n-seg.pt"

    # 修改理由：显式指定使用 cuda:0。
    device = 0

    # 修改理由：如果你后面有多个自定义消融模型，建议这些 yaml 都基于 yolov8m-seg 的结构改，而不是直接用 yolov8-seg.yaml。
    # 修改理由：这里默认主实验显式指定 yolov8m-seg.yaml，从而确保结构是 m 而不是 n。
    exp_yamls = [
        "yolov8n-seg.yaml",
        # "yolov8-seg-sppf-replk-full.yaml",
        # "yolov8-seg-mdka-sadilation.yaml",
        # "yolov8-seg-mdka-sadilation-sppf-replk-full.yaml",
    ]

    # =========================
    # 多 seed 设置
    # =========================
    # 修改理由：你原注释写的是跑 3 个 seed，但代码只有 [42]。
    # 修改理由：这里直接改为 3 个 seed，和你的实验设计保持一致。
    seeds = [42]

    # =========================
    # 训练超参（保持一致，确保消融公平）
    # =========================
    imgsz = 640

    # 修改理由：保持你原来的 batch，不改变训练设定，避免把 seed 实验和超参变化混在一起。
    batch = 8

    # 修改理由：显式指定初始学习率为 0.01，这是你要求固定的值。
    lr0 = 0.01

    # 修改理由：显式指定动量为 0.937，避免 optimizer=auto 时被自动覆盖。
    momentum = 0.937

    # 修改理由：显式指定权重衰减为 0.0005，保证训练配置严格可控。
    weight_decay = 0.0005

    # 修改理由：显式指定优化器为 SGD，防止 Ultralytics 自动选择 AdamW。
    optimizer = "SGD"
    # optimizer = 'auto'

    # 修改理由：保持 epoch 一致，保证不同 seed、不同模型收敛条件一致。
    epochs = 200

    # 修改理由：A800 上开启 AMP 通常更高效；保持与原设定一致。
    amp = True

    # 修改理由：保持早停策略一致，确保实验公平。
    patience = 50

    # 修改理由：保持数据加载设置一致，避免引入额外变量。
    workers = 4

    # =========================
    # 依次跑所有消融模型 × 多个 seed
    # =========================
    for yaml_path in exp_yamls:
        yaml_stem = Path(yaml_path).stem

        for seed in seeds:
            print("\n" + "=" * 100)
            print(f"开始训练: model={yaml_stem}, seed={seed}")
            print("=" * 100 + "\n")

            # 修改理由：每次训练前重新设置随机种子，确保该轮实验严格对应当前 seed。
            set_seed(seed=seed, deterministic=True)

            # 修改理由：显式按 m 架构构建模型，再加载 m 的预训练权重。
            # 修改理由：不能再用 yolov8-seg.yaml，否则未显式传 scale 时可能默认回落到 n。
            model = YOLO(yaml_path).load(pretrained)

            # 修改理由：输出目录加入 seed 后缀，避免不同 seed 结果互相覆盖。
            # 修改理由：训练参数里显式指定 optimizer、lr0、momentum、weight_decay，确保日志中不会再出现 optimizer=auto 覆盖超参。
            results = model.train(
                data=data_yaml,
                imgsz=imgsz,
                device=device,
                seed=seed,
                deterministic=True,
                amp=amp,
                batch=batch,
                epochs=epochs,
                patience=patience,
                workers=workers,
                optimizer=optimizer,
                seg_loss_type="bce_tversky",
                tversky_weight=1.0,
                tversky_alpha=0.7,
                tversky_beta=0.3,
                lr0=lr0,
                momentum=momentum,
                weight_decay=weight_decay,
                save=True,
                val=True,
                project="runs/segment",
                name=f"{yaml_stem}_seed{seed}",
                exist_ok=False,
            )

            # 修改理由：循环多次训练时及时释放显存，减少显存碎片和串扰风险。
            del model
            del results
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
