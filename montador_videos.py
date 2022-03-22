import cv2
import numpy as np
import glob

Numero=len(glob.glob("./data/gif_cocina_avocado/frames/*.jpg"))
print("Número de imágenes en la carpeta:",Numero)
fps = 2
 
img_array = []
for filename in glob.glob('./data/gif_cocina_avocado/frames/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('./data/gif_cocina_avocado.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
print("Done")

