import cv2

vid_capture = cv2.VideoCapture('project.avi')
i = 0

if (vid_capture.isOpened() == False):
	print("Error opening the video file")
	# Read fps and frame count
else:
    while(vid_capture.isOpened()):
    # vCapture.read() methods returns a tuple, first element is a bool
    # and the second is frame
        ret, frame = vid_capture.read()
        if ret == True:
            cv2.imshow('Frame',frame)
	        # 20 is in milliseconds, try to increase the value, say 50 and observe
	        #key = cv2.waitKey(20)
            """
            name = 'resultados/figure_' + str(i)
            cv2.imwrite(name + '.jpg',ret)"""
            
