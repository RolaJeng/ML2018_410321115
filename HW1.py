import numpy as np
import cv2

imgKey1 = cv2.imread("key1.png")
imgKey2 = cv2.imread("key2.png")
imgI = cv2.imread("I.png")
imgE = cv2.imread("E.png")
imgEprime = cv2.imread('Eprime.png')

arrK1 = np.array(imgKey1)
arrK2 = np.array(imgKey2)
arrI = np.array(imgI)
arrE = np.array(imgE)
arrEp = np.array(imgEprime)

outImg = np.zeros(imgEprime.shape, np.uint8)

size = imgKey1.shape
height = size[0]
width = size[1]

epoch = 1
w = np.random.rand(3)
w = np.transpose(w)

maxIterLimit = 2 #max number of epochs
alp = 0.00001 #learning rate

a = np.zeros((height, width))
e = np.zeros((height, width))
x = np.zeros((height, width))
output = np.zeros((height, width))

while (epoch == 1 or epoch < maxIterLimit ):
    for i in range(0, width - 1):
        for j in range(0, height - 1):
            x = [arrK1[j][i], arrK2[j][i], arrI[j][i]]
            x = np.transpose(x)
            w = np.transpose(w)
            a = w * x
            e = arrE[j][i] - a
            w = w + alp * e * x
    epoch = epoch + 1

for i in range(0, width - 1):
    for j in range(0, height - 1):
        output = (arrEp[j][i] - w[0] * arrK1[j][i] - w[1] * arrK2[j][i]) / w[2]
        outImg[j][i] = output

print(w[0], w[1], w[2])

cv2.imshow('Decryption Image', outImg,)
cv2.imwrite('Decryption Image.png', outImg)

cv2.waitKey(0)
cv2.destroyAllWindows()
