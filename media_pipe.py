import cv2
import math
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
#stackOverflow
#https://stackoverflow.com/questions/71094744/mediapipe-crop-images
import dlib
from PIL import Image
from skimage import io

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
        print(img.shape)
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
    #return img
    

mp_face_detection = mp.solutions.face_detection
# Prepare DrawingSpec for drawing the face landmarks later.
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

image_name = 'itsme.jpg'
img_color = cv2.imread(image_name,cv2.IMREAD_COLOR)
print(img_color.shape)
height, width, _ = img_color.shape

#######3
#img_color = resize_and_show(img_color)

print(img_color.shape)

#image size
h, w, c = img_color.shape
print('width:  ', w)
print('height: ', h)

#Detection
with mp_face_detection.FaceDetection(
min_detection_confidence=0.5, model_selection=0) as face_detection:
 
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    face_detection_results = face_detection.process(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))

    # Draw face detections of each face.
    if face_detection_results.detections:
        
        print(face_detection_results.detections)
        annotated_image = img_color.copy()
        for face in (face_detection_results.detections):
            
            # Display the face number upon which we are iterating upon.
            #print(f'FACE NUMBER: {face_no+1}')
            print('---------------------------------')
            
            # Display the face confidence.
            print(f'FACE CONFIDENCE: {round(face.score[0], 2)}')
            
            # Get the face bounding box and face key points coordinates.
            face_data = face.location_data
            print(face.location_data.format)

            #DUDA¡¡
            #b=face_detection_results.detections.location_data
            #a = face.location.relative_bounding_box
            #DUDA¡¡¡¡¡

            # Display the face bounding box coordinates.
            print(f'\nFACE BOUNDING BOX:\n{face_data.relative_bounding_box}')

            
            # Iterate two times as we only want to display first two key points of each detected face.
            for i in range(6):
    
                # Display the found normalized key points.
                print(f'{mp_face_detection.FaceKeyPoint(i).name}:')
                print(f'{face_data.relative_keypoints[mp_face_detection.FaceKeyPoint(i).value]}')

            # Crop the face 
            image_margin = 200
            data = face_data.relative_bounding_box
            plus_wide = 300

            # Write & calculate labels
            with open('labels_media_pipe_video_original.txt', 'a') as f:

                f.write(image_name + ',')
                xleft = data.xmin*w
                xleft = int(xleft) - image_margin - plus_wide
                print(f'Esquina superior, coor X: {xleft}')
                f.write(str(xleft) + ',')

                xtop = data.ymin*h
                xtop = int(xtop) - image_margin -300
                print(f'Esquina superior, coor Y: {xtop}')
                f.write(str(xtop) + ',')

                xright = data.width*w + xleft 
                xright = int(xright) + image_margin +300 + plus_wide
                print(f'Esquina inferior, coor X: {xright}')
                f.write(str(xright) + ',')

                xbottom = (data.height)*h + xtop + image_margin
                xbottom = int(xbottom) + image_margin +300
                print(f'Esquina inferior, coor Y: {xbottom}')
                f.write(str(xbottom) + '\n')


                detected_faces = [(xleft, xtop, xright, xbottom)]
                detected_faces_wr = xleft, xtop, xright, xbottom

                for n, face_rect in enumerate(detected_faces):
                    face = Image.fromarray(annotated_image).crop(face_rect)
                    #name="resultados/frame_"+str(x)    #si le pones png te lo pone 2 veces
                    face.save('prueba.jpeg')
                    face_np = np.asarray(face)
                    plt.imshow(face)
                    plt.show()
                    #plt.savefig('name.png') #save as jpg
                    #plt.imsave('aaaaaaaa',face)

#Mostrar y guardar
plt.imshow(annotated_image)
plt.show()








        
