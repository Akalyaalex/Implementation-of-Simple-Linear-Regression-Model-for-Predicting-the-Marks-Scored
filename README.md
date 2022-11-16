# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

 1. Read the given dataset

 2.Assign values for x and y and plot them

 3.Split the dataset into train and test data

 4.Import linear regression and train the
 data

 5.find Y predict

 6.Plot train and test data

 7.Calculate mse,mae and rmse 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Akalya A
RegisterNumber:  212220220002

##LinearRegression

import pandas as pd
import numpy as np
dataset=pd.read_csv('/content/student_scores.csv')
print(dataset)

# assigning hours to X & Scores to Y
X=dataset.iloc[:,:1].values
Y=dataset.iloc[:,1].values
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='red')
plt.title('Training set (H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show

plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,reg.predict(X_test),color='blue')
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)*/
```

## Output:
![exp 1](https://user-images.githubusercontent.com/114275126/202213583-cd39c8bf-49d8-4e81-a5c3-cd8d509f2d24.PNG)
![exp 2](https://user-images.githubusercontent.com/114275126/202213088-a0472773-6edc-46aa-a92c-3b749899ea62.PNG)
![exp 3](https://user-images.githubusercontent.com/114275126/202213124-4dc692fc-0ffb-4b59-bfa6-0829cb78efe2.PNG)
![exp 4](https://user-images.githubusercontent.com/114275126/202213250-5899637c-0287-497c-96df-070f1896bc9a.PNG)
![exp 5](https://user-images.githubusercontent.com/114275126/202213290-e98925e6-f519-47f5-8945-ed350b9a6335.PNG)
![exp 6](https://user-images.githubusercontent.com/114275126/202213314-ccfcd7c4-43d7-4d11-aaea-cb4924e105aa.PNG)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
