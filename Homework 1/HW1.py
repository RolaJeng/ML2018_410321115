import numpy as np
import cv2

#read the images
imgKey1 = cv2.imread("key1.png")
imgKey2 = cv2.imread("key2.png")
imgI = cv2.imread("I.png")
imgE = cv2.imread("E.png")
imgEprime = cv2.imread('Eprime.png')

#convert images to gray_level
grK1 = cv2.cvtColor(imgKey1, cv2.COLOR_BGR2GRAY)
grK2 = cv2.cvtColor(imgKey2, cv2.COLOR_BGR2GRAY)
grI = cv2.cvtColor(imgI, cv2.COLOR_BGR2GRAY)
grE = cv2.cvtColor(imgE, cv2.COLOR_BGR2GRAY)
grEp = cv2.cvtColor(imgEprime, cv2.COLOR_BGR2GRAY)

#create matrix to the gray images
arrK1 = np.array(grK1)
arrK2 = np.array(grK2)
arrI = np.array(grI)
arrE = np.array(grE)
arrEp = np.array(grEp)
outputImg = np.zeros(arrEp.shape, np.uint8)

#read the size of input image
size = imgKey1.shape
height = size[0]
width = size[1]

#initialize
epoch = 1
w = np.random.rand(3)
x = np.random.rand(3)
e = np.random.rand(3)
maxIterLimit = 10
alp = 0.000001
vig = 0.000001

#tranning algorithm
while (epoch < maxIterLimit and abs(w[1] - w[0]) > vig):
    for i in range(1, width):
        for j in range(1, height):
            tmpK1 = arrK1[j][i]
            tmpK2 = arrK2[j][i]
            tmpI = arrI[j][i]
            x[0] = tmpK1
            x[1] = tmpK2
            x[2] = tmpI
            a = np.dot(w, x)
            e = arrE[j][i] - a
            w = w + alp * e * x
    epoch = epoch + 1

#find the value of w by using the formula
for i in range(1, width):
    for j in range(1, height):
        output = (arrEp[j][i] - w[0] * arrK1[j][i] - w[1] * arrK2[j][i]) / w[2]
        if np.any(output > 255):
            output = 255
        if np.any(output < 0):
            output = 0
        outputImg[j][i] = output

print(w)
cv2.imwrite('Output Image.png', outputImg)
cv2.imshow('Output Image.png', outputImg)
cv2.waitKey(0)

