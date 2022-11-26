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
dataset=pd.read_csv('/content/Placement_Data.csv')
dataset.head()
dataset.tail()
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
print(X)
print(Y)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='orange')
plt.title('Training set (H vs S)')
plt.xlabel('Hours')
plt.ylabel("scores")
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,reg.predict(X_test),color='black')
plt.title('Test set (H vs S)')
plt.xlabel('Hours')
plt.ylabel("scores")
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE =  ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

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
print("RMSE = ",rmse)
*/
```


## Output:
![1](https://user-images.githubusercontent.com/114275126/204101071-184e5b14-dc5f-47c0-9656-99d67454b74f.PNG)
![2](https://user-images.githubusercontent.com/114275126/204101119-3701aab7-8e92-4d4d-9d47-b3a9aa4263dd.PNG)
![33](https://user-images.githubusercontent.com/114275126/204101163-06d4661f-0f18-4604-a1a2-834e5e8d1095.PNG)
![4](https://user-images.githubusercontent.com/114275126/204101195-ea5b529f-c6fa-4e4a-bfa4-e417b446db10.PNG)
## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
