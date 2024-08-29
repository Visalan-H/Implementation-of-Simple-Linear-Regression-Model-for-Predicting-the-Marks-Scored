# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```python
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Visalan H
RegisterNumber: 212223240183
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error,mean_squared_error
#read csv file
df=pd.read_csv('student_scores.csv')
print(df.head())
print(df.tail())
# Segregating data to variables
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
#import linear regression model and fit the model with the data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
#displaying predicted values
print("Visalan H")
print("212223240183")
print(y_pred)
#displaying actual values
print("Visalan H")
print("212223240183")
y_test
#graph plot for training data
print("Visalan H")
print("212223240183")
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
print("Visalan H")
print("212223240183")
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#find mae,mse,rmse
print("Visalan H")
print("212223240183")
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
### df.head
![image](https://github.com/user-attachments/assets/9d934be1-75f0-41da-9d4e-3f0bd03ad443)
### df.tail()
![image](https://github.com/user-attachments/assets/167e5b46-008b-4c84-94ac-762ce3993155)
### Array value of X
![image](https://github.com/user-attachments/assets/b14ddb08-33e3-42b0-ab5c-24440fb3e290)
### Array value of Y
![image](https://github.com/user-attachments/assets/789feb99-9417-4c7a-94af-594a66bfe3d0)
### Values of Y prediction
![image](https://github.com/user-attachments/assets/23447eb4-e822-45d8-a229-9ef6909b1e9e)
### Array values of Y test
![image](https://github.com/user-attachments/assets/faaad233-fe07-4cae-9810-c63282dc5ef2)
### Training Set Graph
<img src="https://github.com/user-attachments/assets/e0cb1e69-1b88-4fb2-96c6-10ffbb1f7333" width="300"/>
### Test Set Graph
![image](https://github.com/user-attachments/assets/0f10af88-1007-48aa-b9ed-ce9e68046684)
### Values of MSE, MAE and RMSE
![image](https://github.com/user-attachments/assets/c2435d3b-e61f-4b3c-8805-c45ea3679c3b)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
