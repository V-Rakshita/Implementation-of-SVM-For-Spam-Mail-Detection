# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries
2. Read the dataset and separate the independent and dependent variables.
3. Split the given data into training and testing data.
4. For preprocessing, using CountVectorizer.
5. Train the model using SVC() and fit it by passing the training data.
6. predict the model by passing the testing data.
7. measure the accuracy score.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..  
*/

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
data = pd.read_csv("spam.csv",encoding="Windows-1252")
data.info()
x = data['v2'].values
y = data['v1'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)
cv = CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test = cv.transform(x_test)
svm_model=SVC()
svm_model.fit(x_train,y_train)
y_pred = svm_model.predict(x_test)
acs = accuracy_score(y_test,y_pred)
print("Accuracy score is: ",acs)
```

## Output:
![image](https://github.com/user-attachments/assets/90149c1c-f6b6-485c-991b-354630426581)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
