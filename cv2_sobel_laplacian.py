import cv2
import numpy as np
import sys

def camera():
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        # lapacian(frame)
        # k = cv2.waitKey(1)
        # if k == ord('q'):
            # continue
        sobel(frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

def lapacian(img): 
	

	if img is None:
		sys.exit("無法讀取影像...")

	#laplacian
	laplacian = cv2.Laplacian(img,cv2.CV_64F)

	print(laplacian.shape)
	cv2.imshow('laplacian',laplacian)
	cv2.imshow('src',img)
	

def sobel(img): 
	

	if img is None:
		sys.exit("無法讀取影像...")
	#sobelx
	sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
	#sobely
	sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

def main(argv=None):
    if argv is None:
        argv=sys.argv
    print(argv)
    print('OpenCV 版本:',cv2.__version__)
    #影像梯度計算，色彩或強度方向性變化
    camera()
    #lapacian
    #Sobel

if __name__ == '__main__':
    sys.exit(main())
