from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

#digit recognition output images label
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
def naiveBayes(trainData, testData, trainTarget, testTarget):
    nb_clf = MultinomialNB()
    nb_clf.fit(trainData, trainTarget)
    nb_acc = nb_clf.score(testData, testTarget)
    nb_acc = "%.4f" % nb_acc
    print("Accuracy of Naive Bayes: " + str(nb_acc))
    predictions = nb_clf.predict(testData)
    print(classification_report(testTarget, predictions))
    p = np.random.permutation(len(testData))
    p = p[:20]
    pltImages(testData[p].reshape(-1, 28, 28), predictions[p])

#train model using SVM
def SVM(trainData, testData, trainTarget, testTarget):
    svm_clf = svm.SVC()
    svm_clf.fit(trainData, trainTarget)
    svm_acc = svm_clf.score(testData, testTarget)
    svm_acc = "%.4f" % svm_acc
    print("Accuracy of SVM: " + str(svm_acc))
    predictions = svm_clf.predict(testData)
    print(classification_report(testTarget, predictions))
    p = np.random.permutation(len(testData))
    p = p[:20]
    pltImages(testData[p].reshape(-1, 28, 28), predictions[p])

#train model using Nearest Neighbors
def NN(trainData, testData, trainTarget, testTarget):
    nn_clf = KNeighborsClassifier()
    nn_clf.fit(trainData, trainTarget)
    nn_acc = nn_clf.score(testData, testTarget)
    nn_acc = "%.4f" % nn_acc
    print("Accuracy of Nearest Neighbors: " + str(nn_acc))
    predictions = nn_clf.predict(testData)
    print(classification_report(testTarget, predictions))
    p = np.random.permutation(len(testData))
    p = p[:20]
    pltImages(testData[p].reshape(-1, 28, 28), predictions[p])

mnist = fetch_mldata('MNIST original')
mnData = mnist['data']
mnTarget = mnist['target']
mnTarget = mnTarget.astype("int32")
mnData = mnData / 255.0 #Normalization
mnData.min(), mnData.max()
trainData, testData, trainTarget, testTarget = train_test_split(mnData, mnTarget)
trainData.shape, testData.shape

#naiveBayes(trainData, testData, trainTarget, testTarget)
#SVM(trainData, testData, trainTarget, testTarget)
NN(trainData, testData, trainTarget, testTarget)

