import cv2
import numpy as np
import utils


webCamFeed = True
cap = cv2.VideoCapture(0)
cap.set(10,160)
heightImg = 640
widthImg  = 480


count=0
imgAdaptiveThre = None
while True:

    if webCamFeed:success, img = cap.read()
    else:
        img = cv2.imread('1.jpg')
    img = cv2.resize(img, (widthImg, heightImg))
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) 
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) 
    imgThreshold = cv2.Canny(imgBlur, 100, 200)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) 
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)
    imgContours = img.copy() 
    imgBigContour = img.copy() 
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) 

    biggest, maxArea = utils.biggestContour(contours)
    print(biggest)
    if biggest.size != 0:
        biggest=utils.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
        imgBigContour = utils.drawRectangle(imgBigContour,biggest,2)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        cv2.imshow('Warped Document', imgWarpColored)

    cv2.imshow('Contours', imgBigContour)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
