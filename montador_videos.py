import os
import cv2
import glob

carpeta = "/home/alvaro.nieva/Documents/nieva_TFM/data/TUM_KITCHEN/cam3"
fps = 25
image_list = os.listdir(carpeta)
image_list.sort()

img_array = []
for filename in image_list:
    name = carpeta + '/' + filename
    img = cv2.imread(name)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('./data/tmktch_cam3.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
print("Done")

