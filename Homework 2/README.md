## ML2018_410321115
### Assignment 2: Handwritten Digit Recognition
 
### Classifier  
___
我使用三種分類器: _Naive Bayes, SVM, Nearest Neighbors_  
  
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
![Alt text](https://github.com/RolaJeng/ML2018_410321115/blob/master/Homework%202/nb%20result.JPG)  
準確率:  
![Alt text](https://github.com/RolaJeng/ML2018_410321115/blob/master/Homework%202/nb%20acc.JPG)  

#### _Nearest Neighbors:_  
```python
nn_clf = KNeighborsClassifier()    
nn_clf.fit(trainData, trainTarget)  
```  

結果圖:  
![Alt text](https://github.com/RolaJeng/ML2018_410321115/blob/master/Homework%202/nb%20result.JPG)  
準確率:  
![Alt text](https://github.com/RolaJeng/ML2018_410321115/blob/master/Homework%202/nb%20acc.JPG)  
 
### Evaluating the performance of the classifier  
___  

準確率比較:  

Classifier | Accuracy
:----------: | :--------:
Naive Bayes | 0.83
SVM | 0.96
Nearest Neighbors|0.95


#### Dataset
sklearn可以下載內建的fetch data訓練資料  

```python
mnist = fetch_mldata('MNIST original')  
```
#### Normalization



