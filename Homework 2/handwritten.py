from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn import svm
import time

mnist = fetch_mldata('MNIST original')
mnData = mnist['data']
mnTarget = mnist['target']

mnTarget = mnTarget.astype("int32")
mnData = mnData / 255.0
mnData.min(), mnData.max()

trainData, testData, trainTarget, testTarget = train_test_split(mnData, mnTarget)
trainData.shape, testData.shape
#print(mnData.dtype, mnTarget.dtype)
#print(mnData.shape, mnTarget.shape)

def pltImages(images, labels):
    cols = min(5, len(images))
    rows = len(images) // cols
    fig = plt.figure(figsize=(8, 8))

    for i in range(rows * cols):
        subplot = fig.add_subplot(rows, cols, i+1)
        plt.axis("off")
        plt.imshow(images[i], cmap=plt.cm.gray)
        subplot.set_title(labels[i])
    plt.show()

#train model using Naive Bayes
#cls = MultinomialNB()
#cls.fit(trainData, trainTarget)
#evaluate model
#cls.score(testData, testTarget)

svm_clf = svm.SVC()
svm_clf.fit(trainData, trainTarget)
svm_clf.score(testData, testTarget)
predictions = svm_clf.predict(testData)

#predictions = cls.predict(testData)

print(classification_report(testTarget, predictions))

p = np.random.permutation(len(testData))
p = p[:20]
pltImages(testData[p].reshape(-1, 28, 28), predictions[p])
