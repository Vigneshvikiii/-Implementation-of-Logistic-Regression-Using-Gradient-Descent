# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
~~~
1.Import the packages required.
2.Read the dataset.
3.Define X and Y array.
4.Define a function for costFunction,cost and gradient. 
5.Define a function to plot the decision boundary and predict the Regression value. 
~~~

## Program:
```

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Vignesh S
RegisterNumber: 212223230240

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('Placement_Data_Full_Class.csv')
dataset.head()

dataset.info()

dataset = dataset.drop('sl_no', axis=1);
dataset.info()

dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

x = dataset.iloc[:,:-1]
x

y=dataset.iloc[:,-1]
y

import numpy as np
theta = np.random.rand(x.shape[1])
y=y

def sigmoid(z):
  return 1/(1+np.exp(-z))

def loss(theta, X, y):
  h = sigmoid(X.dot(theta))
  return npm.sum(y*np.log(h) + (1-y)*np.log(1-h))

def gradient_descent(X, y, theta, alpha, iterations):
  m = len(y)
  for i in range(iterations):
    h = sigmoid(X.dot(theta))
    gradient = X.T.dot(h-y) / m
    theta -= alpha * gradient
  return theta

theta = gradient_descent(x, y, theta, 0.01, 1000)

def predict(theta, X):
  h = sigmoid(X.dot(theta))
  y_pred = np.where(h >= 0.5, 1, 0)
  return y_pred 

y_pred = predict(theta, x)

accuracy = np.mean(y_pred == y)
print("Accuracy:", accuracy)
print("Predicted:\n",y_pred)
print("Actual:\n",y.values)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print("Predicted Result:",y_prednew)
```

## Output:
## Head 
![image](https://github.com/user-attachments/assets/14ba0389-6850-4296-b2ca-7635dd1204b1)

## Info 

![image](https://github.com/user-attachments/assets/fe852175-32c2-4160-b1c3-1b0451ad18b8)

## INFO 
![image](https://github.com/user-attachments/assets/e6bba073-e008-4c16-af72-cef4e1c00657)
## Changing into category:
![image](https://github.com/user-attachments/assets/08f566ba-573d-44eb-bcd4-2b5810e9e431)
## Changing into category codes:
![image](https://github.com/user-attachments/assets/80609a01-b246-4080-b1ec-8244985792bc)
## Value of X:
![image](https://github.com/user-attachments/assets/7a2acfe9-1366-4dff-9f2c-92893f9dd917)
## Value of Y:
![image](https://github.com/user-attachments/assets/1871e11d-ee78-4a50-818c-330194e08cf9)
## Predicted Value:
![image](https://github.com/user-attachments/assets/7c57c0d0-95ea-46d2-ab6c-6932ac9bd7ce)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

