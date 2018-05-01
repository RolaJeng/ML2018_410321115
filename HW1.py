import numpy as np
import cv2

epoch = 1

imgKey1 = cv2.imread("key1.png")
imgKey2 = cv2.imread("key2.png")
imgI = cv2.imread("I.png")
imgE = cv2.imread("E.png")
imgEprime = cv2.imread('Eprime.png')

#initial Weights of w
w = np.array([1, 1, 1])
w.transpose()

#height and width of Image
size = imgKey1.shape

height = size[0]
width = size[1]
pixelsValue = size[2]

limit = 2
alpha = 0.00001
#initial value of matrix
arrayA = np.zeros(0)
arrayE = np.zeros(0)
arrayI = np.zeros(0)

while (epoch < limit ):
    for i in range(1, height):
        for j in range(1, width):
            arrayA = np.transpose(w) * np.transpose([imgKey1, imgKey2, imgI])
            arrayE = imgE - arrayA
            w = w + alpha * arrayE * np.transpose([imgKey1, imgKey2, imgI])
    epoch = epoch + 1

#for i in range(1, width):
   # for j in range(1, height):
    #    arrayI = imgEprime - w(1) * imgKey1 - w(2) * imgKey2 / w(3)

#cv2.imshow(imgI)
