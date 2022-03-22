import cv2


i = 0
name = './data/pruebas/figure_'
tipo = '.jpg'
# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv2.VideoCapture('./data/gif_cocina_avocado/gif_cocina_avocado.avi') 
if (vid_capture.isOpened() == False):
    print("Error opening the video file")
# Read fps and frame count
else:
    while(vid_capture.isOpened()):
        # vid_capture.read() methods returns a tuple, first element is a bool
        # and the second is frame
        ret, frame = vid_capture.read()
        if ret == True:
            cv2.imshow('Frame',frame)
            cv2.setWindowProperty('frame', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
            # 20 is in milliseconds, try to increase the value, say 50 and observe
            key = cv2.waitKey(50)
            i +=1
            filename = name + str(i) + tipo
            cv2.imwrite(filename, frame)
        else:
            break

# Release the video capture object
print("Done. Writen images at:",name)
vid_capture.release()
cv2.destroyAllWindows()
