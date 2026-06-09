# Librería para lectura/escritura de imágenes y manipulación de pixels
import cv2
# Librería para operaciones matemáticas y manejo de arrays numéricos
import numpy as np
# API principal de Depth Anything 3 para carga de modelos e inferencia
from depth_anything_3.api import DepthAnything3

# Cargar modelo Small desde HuggingFace (0.08B parámetros, Apache 2.0)
# from_pretrained descarga los pesos automáticamente la primera vez
model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
# eval() desactiva dropout/batchnorm; cuda() mueve el modelo a la GPU
model.eval().cuda()

# Leer imagen en disco (BGR por defecto en OpenCV)
image = cv2.imread("bin_sandias.png")
# Convertir de BGR (OpenCV) a RGB (estándar para modelos de visión)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Ejecutar inferencia: recibe lista de imágenes numpy RGB
# Retorna un objeto Prediction con depth, extrinsics, intrinsics, etc.
prediction = model.inference([image_rgb])
