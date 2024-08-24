from ultralytics import YOLO


model = YOLO.load("/path/last.pt")  # build from YAML and transfer weights
 
# Train the model
# 提示参数：epochs, imgsz, batch,resume
# data 为 yolo 数据配置
results = model.train(
    data="/home/ganxin/fa/ais/workspace/questionC/yoloTrain/dataset/datasets.yaml",
)