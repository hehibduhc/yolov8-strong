# train.py
from ultralytics import YOLO

def main():
    # 1. 加载YOLO8-seg模型（轻量版n、标准版s、大模型l/x可选）
    # 方式1：先加载yaml配置再加载预训练权重（推荐，可自定义网络）
    model = YOLO('yolov8-seg.yaml').load('yolov8n-seg.pt')  
    # 方式2：直接加载预训练权重（更简洁，无需手动指定yaml）
    # model = YOLO('yolo11n-seg.pt')  

    # 2. 训练分割模型（参数可按需调整）
    results = model.train(
        data="ultralytics/cfg/datasets/crack_seg.yaml",  # 关键：替换为分割任务的数据集配置
        amp=False,  # 保留禁用半精度（若训练不稳定可关闭）
        # 可选：添加分割任务常用参数
        batch=8,  # 自动适配批次大小（根据GPU显存）
        patience=50,  # 早停耐心值（防止过拟合）
        save=True,  # 保存最佳模型
        val=True,  # 训练中验证
        epochs = 200
        
    )

if __name__ == '__main__':
    main()