## ML2018_410321115
## Assignment 2: Handwritten Digit Recognition
 
### ● Classifier  
___
我使用三種分類器: _Naive Bayes, SVM, K-Nearest Neighbors_  
  
#### _Naive Bayes:_  
```python
nb_clf = MultinomialNB()  
nb_clf.fit(trainData, trainTarget)    
```  

結果圖:  
![Alt text](https://github.com/RolaJeng/ML2018_410321115/blob/master/Homework%202/nb%20result.JPG)
準確率:  
![Alt text](https://github.com/RolaJeng/ML2018_410321115/blob/master/Homework%202/nb%20acc.JPG)  
  
#### _SVM:_   
```python
svm_clf = svm.SVC()  
svm_clf.fit(trainData, trainTarget)  
```  

結果圖:  
![Alt text](https://github.com/RolaJeng/ML2018_410321115/blob/master/Homework%202/svm%20result.JPG)  
準確率:  
![Alt text](https://github.com/RolaJeng/ML2018_410321115/blob/master/Homework%202/svm%20acc.JPG)  

#### _K-Nearest Neighbors:_  
```python
nn_clf = KNeighborsClassifier()    
nn_clf.fit(trainData, trainTarget)  
```  

結果圖:  
![Alt text](https://github.com/RolaJeng/ML2018_410321115/blob/master/Homework%202/knn%20result.JPG)  
準確率:  
![Alt text](https://github.com/RolaJeng/ML2018_410321115/blob/master/Homework%202/knn%20acc.JPG)  
 
### ● Evaluating the performance of the classifier  
___  

準確率比較:  

Classifier | Accuracy
:----------: | :--------:
Naive Bayes | 0.83
SVM | 0.94
K-Nearest Neighbors|0.95

### ● Steps   
___  

#### Download Dataset
sklearn可以下載內建的fetch data訓練資料  

```python
mnist = fetch_mldata('MNIST original')  
```
#### Normalization 

```python
mnData = mnist['data'] 
mnData = mnData / 255.0
```  

#### Test data and Train data 
隨機劃分測試資料以及訓練資料  

```python
trainData, testData, trainTarget, testTarget = train_test_split(mnData, mnTarget)
```

#### sklearn Machine Learning functions 
套用演算法functions導入訓練資料以及測試資料  

```python
naiveBayes(trainData, testData, trainTarget, testTarget)  
SVM(trainData, testData, trainTarget, testTarget)  
KNN(trainData, testData, trainTarget, testTarget)  
```  
#### 列出辨識的隨機5X4張圖片 
挑出辨識結果的隨機20張圖片並顯示測試結果  

```python
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
```  
### ● Discussion  
___  

這次的作業其實沒有想像中的困難，利用老師上課教的演算法以及自己上網找的演算法，都在sklearn中可以找到相對應的程式庫，所以其實很快速地就能套用完，只是發現有些演算法需要花費相當多的時間，因此在跑程式的過程，等待了好一陣子，以為是電腦當機，最後還是順利跑完。  
我使用了三種分類器，可以發現Naive Bayes、SVM和K-Nearest Neighbors當中，最準確的是SVM，而最不精準的是Naive Bayes。  
sklearn的分類器有很多種，我們可以再多多發掘不同的分類器去測試，透過實驗找出自己需要的最佳演算法。  
謝謝機器學習導論這堂課讓我學習python以及機器學習的知識。  
