from roboflow import Roboflow

rf = Roboflow(api_key="Inysi7LxNzDNzc85mi0k")  # 替换为授权后的密钥
project = rf.workspace("potcasdace").project("single-crack")
version = project.version(1)
dataset = version.download("yolov8")  # 下载为 YOLOv8 兼容格式
