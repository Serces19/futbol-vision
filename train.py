
import numpy as np
from ultralytics import YOLO
from roboflow import Roboflow


rf = Roboflow(api_key="RLWs7gxhN3B6JO7kuPob")
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(20)
dataset = version.download("yolov8")

# project = rf.workspace("sergiotest-lwmd3").project("vision-fqr1p")
# version = project.version(2)
# dataset = version.download("yolov8")


# Carga tu modelo pre-entrenado. Si es un modelo de Ultralytics, puedes usar el nombre.
# Si es un archivo .pt que ya tienes, reemplaza 'yolov8n.pt' por 'tu_modelo.pt'.
model = YOLO('./models/yolov8m-football.pt') 

# Paso 4 & 5: Configuración y ejecución del fine-tuning
# El path al archivo data.yaml
data_path = f'{dataset.location}/data.yaml'
    
# Inicia el entrenamiento (fine-tuning)
results = model.train(
    data=data_path,
    epochs=100,
    imgsz=1280,
    batch=8,
    patience=20,
    project='yolo_fine_tuning',
    name='football_detection_model'
)

# Opcional: Evaluar el modelo final en el conjunto de validación
metrics = model.val()
print(metrics)

# Opcional: Exportar el modelo entrenado
model.export(format='onnx')

