from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_mldata('MNIST original')
mnData = mnist['data']
mnTarget = mnist['target']

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

p = np.random.permutation(len(mnData))
p = p[:20]
pltImages(mnData[p].reshape(-1, 28, 28), mnTarget[p])

#img = np.reshape(mnist.data[0, :], (28, 28))
#plt.imshow(img, cmap = plt.cm.gray)
#plt.show()