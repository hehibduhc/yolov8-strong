import random
import shutil
from pathlib import Path

from tqdm import tqdm  # 进度条库，提升用户体验


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_seed: int = 42,  # 固定随机种子保证划分结果可复现
):
    """将图片和对应的标签文件按比例划分为训练集、验证集、测试集.

    Args:
        source_dir: 源数据目录，需包含 'images' 和 'labels' 子文件夹
        output_dir: 输出目录，会创建划分后的目录结构
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
    """
    # 校验比例之和是否为1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("训练集、验证集、测试集比例之和必须为1！")

    # 设置随机种子保证结果可复现
    random.seed(random_seed)

    # 定义路径
    source_images = Path(source_dir) / "images"
    source_labels = Path(source_dir) / "labels"
    output_dir = Path(output_dir)

    # 校验源目录是否存在
    if not source_images.exists():
        raise FileNotFoundError(f"图片目录不存在: {source_images}")
    if not source_labels.exists():
        raise FileNotFoundError(f"标签目录不存在: {source_labels}")

    # 创建输出目录结构
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件（支持常见图片格式）
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = [f for f in source_images.iterdir() if f.suffix.lower() in image_extensions and f.is_file()]

    if not image_files:
        raise ValueError("未找到任何图片文件！")

    # 过滤出有对应标签的图片
    valid_files = []
    for img_file in image_files:
        label_file = source_labels / f"{img_file.stem}.txt"
        if label_file.exists():
            valid_files.append(img_file)
        else:
            print(f"警告: 图片 {img_file.name} 无对应标签文件，已跳过")

    if not valid_files:
        raise ValueError("没有找到同时包含图片和标签的有效文件！")

    # 打乱文件列表
    random.shuffle(valid_files)

    # 计算划分数量
    total = len(valid_files)
    train_num = int(total * train_ratio)
    val_num = int(total * val_ratio)
    # 测试集数量 = 总数 - 训练集 - 验证集（处理小数精度问题）
    total - train_num - val_num

    # 划分文件
    train_files = valid_files[:train_num]
    val_files = valid_files[train_num : train_num + val_num]
    test_files = valid_files[train_num + val_num :]

    print("\n数据集划分完成：")
    print(f"总有效文件数: {total}")
    print(f"训练集: {len(train_files)} 个文件 ({len(train_files) / total:.1%})")
    print(f"验证集: {len(val_files)} 个文件 ({len(val_files) / total:.1%})")
    print(f"测试集: {len(test_files)} 个文件 ({len(test_files) / total:.1%})")

    # 复制文件到对应目录
    def copy_files(file_list, split):
        """复制图片和标签到指定划分目录."""
        for img_file in tqdm(file_list, desc=f"复制{split}集文件"):
            # 复制图片
            dst_img = output_dir / "images" / split / img_file.name
            shutil.copy2(img_file, dst_img)

            # 复制对应标签
            label_file = source_labels / f"{img_file.stem}.txt"
            dst_label = output_dir / "labels" / split / label_file.name
            shutil.copy2(label_file, dst_label)

    # 执行复制
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

    print(f"\n✅ 数据集划分完成！输出目录: {output_dir.absolute()}")
    print("目录结构：")
    print(f"├── {output_dir.name}/")
    print("│   ├── images/")
    print("│   │   ├── train/")
    print("│   │   ├── val/")
    print("│   │   └── test/")
    print("│   └── labels/")
    print("│       ├── train/")
    print("│       ├── val/")
    print("│       └── test/")


if __name__ == "__main__":
    # ===================== 配置参数 =====================
    # 源数据目录（需包含 images 和 labels 子文件夹）
    SOURCE_DIRECTORY = "D:/BaiduNetdiskDownload/crack-seg"  # 你的train文件夹路径
    # 输出目录（划分后的数据集存放位置）
    OUTPUT_DIRECTORY = "D:/BaiduNetdiskDownload/crack-seg-paper-stadrad"
    # 划分比例（6:2:2）
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2
    # ====================================================

    try:
        split_dataset(
            source_dir=SOURCE_DIRECTORY,
            output_dir=OUTPUT_DIRECTORY,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
        )
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
        exit(1)
