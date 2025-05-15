!pip install ultralytics
!pip install roboflow

from ultralytics import YOLO
from roboflow import Roboflow
rf = Roboflow(api_key="DhuHVzXZTZAPchTSjPcj")
project = rf.workspace("2-gjmol").project("sar-jordan")
version = project.version(11)
dataset = version.download("yolov8")

model=YOLO('/path/YOLO8Xvisdron.pt')
train_results=model.train(
     model=model,
     project=("SAR in Jordan"),
     data="/content/SAR-Jordan-11/data.yaml",
     epochs = 100,
     imgsz=1280,
     patience=15,
     save=True,
     save_period=1,
     pretrained=True,
     plots=True,
     batch=0.8,
     verbose=True,
     val=True,

)
