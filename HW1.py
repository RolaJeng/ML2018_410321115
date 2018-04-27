import numpy as np
import cv2

epoch = 1

imgKey1 = cv2.imread("key1.png")
imgKey2 = cv2.imread("key2.png")
imgI = cv2.imread("I.png")
imgE = cv2.imread("E.png")
imgEprime = cv2.imread('Eprime.png')

#設定初始權重
output = [1, 1, 1]
#圖片的長寬
height, width, channels = imgKey1.shape

#將每個pixel照演算法的公式去做推理W的動作
arrayA = np.zeros((height, width, 3), np.uint8)
arrayE = np.zeros((height, width, 3), np.uint8)
arrayI = np.zeros((height, width, 3), np.uint8)
r = 0.00001

while (epoch < 2):
    for i in range(1, width):
        for j in range(1, height):
            arrayA = output






