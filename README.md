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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("/content/student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values #assigning colum hours to X
X  
Y=dataset.iloc[:,1].values 
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_train,regressor.predict(X_train),color="yellow")
plt.title("Hours vs Scores(Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
*/
```


## Output:
![1](https://user-images.githubusercontent.com/114275126/204101775-ce2f1fca-3776-4bba-83dd-cc1b1098f7fe.jpeg)
![2](https://user-images.githubusercontent.com/114275126/204101827-545ad42f-88f4-4ad1-b01d-009245470109.jpeg)
![3](https://user-images.githubusercontent.com/114275126/204101819-7e8800c6-e32a-42b7-bd42-6084e75e0cd0.jpeg)
![4](https://user-images.githubusercontent.com/114275126/204101801-d2ee0322-d9ab-46c5-8177-3eac0106fa75.jpeg)
![5](https://user-images.githubusercontent.com/114275126/204101843-3782c792-a6a5-457a-8f22-c23827a7c79d.jpeg)
![6](https://user-images.githubusercontent.com/114275126/204101850-736e9c41-d29c-49d8-a7c1-da65ab29905a.jpeg)
![7](https://user-images.githubusercontent.com/114275126/204101857-11d270bf-a9b9-4103-a79a-2e64f9be1a5f.jpeg)
![8](https://user-images.githubusercontent.com/114275126/204101860-33826e96-d871-4b63-921c-e83143bdd5ac.jpeg)
![9](https://user-images.githubusercontent.com/114275126/204101892-f5a8d259-c718-480d-887d-78961cdae127.jpeg)

## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
