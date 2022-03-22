import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import backend as K
from keras.activations import softmax

def softmax_image(x):
    """
    Softmax activation function for each probability map
    """
    return softmax(x, axis=2)  # (?, 98, 65536)

def load_custom_objects(train_mode):
    if train_mode == 'lnds' or train_mode == 'multitask':
        return {'softmax_image': softmax_image}
    else:
        return {}

## Load trained model
## Multitask trained model for landmarks en nuestra revista de Pattern Recognition Letters 2019.
from google.colab import drive
drive.mount('/content/drive/')

# Leer la red de estimación de landmarks el PRL 2019 en formato tensorflow desde OpenCV 
model_landmarks = keras.models.load_model('/content/drive/My Drive/FRANK/landmarks_prl19_PCR/wflw_prl19.hdf5', custom_objects=load_custom_objects('lnds'))
model_landmarks.summary()

# Sacado de: https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
# Detector SSD (Single Shot Face Detector) en OpenCV   
modelFile = "/content/drive/My Drive/FRANK/landmarks_prl19_PCR/opencv_face_detector_uint8.pb"
configFile = "/content/drive/My Drive/FRANK/landmarks_prl19_PCR/opencv_face_detector.pbtxt"
face_detector_net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

# Detectar caras:
file_name = "/content/drive/My Drive/FRANK/landmarks_prl19_PCR/imagenes/Equipo-de-Trabajo-1200x545_c.jpg"
#file_name = "/content/drive/My Drive/FRANK/headpose_estimation_PCR/imagenes/seq_001794.jpg"
image = cv2.imread(file_name)
image_plot = image.copy()

(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
 
face_detector_net.setInput(blob)
detections = face_detector_net.forward()
bboxes = []
conf_threshold = 0.2 # Muy bajo para que detecte todas las caras!! (con 0.7 detecta 2 caras)
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > conf_threshold:
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)

        bboxes.append([x1, y1, x2-x1+1, y2-y1+1])
        cv2.rectangle(image_plot, (x1, y1), (x2, y2), (0, 255, 0), int(round(h/150)), 8)

#bboxes.append([129, 67, 12, 15])
#cv2.rectangle(image_plot, (129, 67), (141, 82), (0, 255, 0), int(round(h/150)), 8)
plt.imshow(cv2.cvtColor(image_plot, cv2.COLOR_BGR2RGB))
          


