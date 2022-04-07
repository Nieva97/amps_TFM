import os
import cv2 as cv
import math
import numpy as np
import pandas as pd
from PIL import Image
import mediapipe as mp
import matplotlib.pyplot as plt

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480


def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
        print(img.shape)
        cv.imshow("Display window", image)
        k = cv.waitKey(0)
    return img


# Inicialización de MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Lectura de imagenes from:
# image_dir = 'data/gif_cocina_avocado/frames'
image_dir = 'data/TUM_KITCHEN/cam2'
# image_dir = 'data'
images = os.listdir(image_dir)
image_list = os.listdir(image_dir)
image_list.sort()

# Figura
fig = plt.figure()
det = 0

for i in image_list:
    # frame_raw = Image.open(os.path.join(image_dir, i))
    # frame_raw = frame_raw.convert('RGB')
    image = cv.imread(os.path.join(image_dir, i))
    # cv.imshow("Display window", frame_raw)
    # k = cv.waitKey(0)
    print(i)

    width, height, info = image.shape
    # image = np.array(frame_raw)
    image = resize_and_show(image)

    # Mostrar
    plt.clf()
    # clear figure: asi no va dando saltos la imagen

    with mp_face_detection.FaceDetection(min_detection_confidence=0.4, model_selection=0) as face_detection:
        face_detection_results = face_detection.process(image)
        # Draw face detections of each face.
        if face_detection_results.detections:
            annotated_image = image.copy()
            for face in (face_detection_results.detections):
                # Display the face number upon which we are iterating upon.
                print('---------------------------------')

                # Display the face confidence.
                print(f'FACE CONFIDENCE: {round(face.score[0], 2)}')

                # Draw the Faces
                mp_drawing.draw_detection(annotated_image, face)

                # Get the face bounding box and face key points coordinates.
                face_data = face.location_data

                # Display the face bounding box coordinates.
                # print(f'\nFACE BOUNDING BOX:\n{face_data.relative_bounding_box}')

                # Iterate two times as we only want to display first two key points of each detected face.
                """for i in range(6):
                    # Display the found normalized key points.
                    print(f'{mp_face_detection.FaceKeyPoint(i).name}:')
                    print(f'{face_data.relative_keypoints[mp_face_detection.FaceKeyPoint(i).value]}')"""

                # Write & calculate labels
                data = face_data.relative_bounding_box

                # Esquina superior, coor X
                xleft_top = int(data.xmin * width)
                # xleft_top = int(xleft_top)
                print(f'Esquina superior, coor X: {xleft_top}')

                # Esquina superior, coor Y
                yleft_top = int(data.ymin * height)
                # yleft_top = int(yleft_top)
                print(f'Esquina superior, coor Y: {yleft_top}')

                # Esquina inferior, coor X
                xright_bot = int(data.width * width + xleft_top)
                # xright_bot = int(xright_bot)
                print(f'Esquina inferior, coor X: {xright_bot}')

                # Esquina inferior, coor Y
                yright_bot = int(data.height * height + yleft_top)
                # yright_bot = int(yright_bot)
                print(f'Esquina inferior, coor Y: {yright_bot}')

                detected_faces = dict(frame=i, left=xleft_top, top=yleft_top, right=xright_bot, bottom=yright_bot)

                row = image_list.index(i)
                df = pd.DataFrame({
                        "frame": i,
                        "left": xleft_top,
                        "top": yleft_top,
                        "right": xright_bot,
                        "bottom": yright_bot,
                    }, index=[row])

                df.to_csv("prueba_etiquetas.txt", header=False, sep=",", index=False, mode='a')

                for n, face_rect in enumerate(detected_faces):
                    crop_image = np.array(image)
                    recorte_x = xright_bot - xleft_top
                    recorte_y = yright_bot - yleft_top
                    crop_image = crop_image[yleft_top:yleft_top + recorte_y, xleft_top:xleft_top + recorte_x]
                    """plt.imshow(crop_image)
                    plt.show()"""

    cv.imshow("Display window", image)






