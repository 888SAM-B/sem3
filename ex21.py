import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
datainit=load_diabetes()

#print(datainit)
data=pd.DataFrame(datainit.data, columns=datainit.feature_names)

data["target"]=datainit.target

X=data[['bmi']]
y=data['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)
prediction=model.predict(X_test)

mse=mean_squared_error(y_test,prediction)
rmse=np.sqrt(mse)

r2s=r2_score(y_test,prediction)
print("Simple Linear Regression")
print("MSE :", mse)
print("RMSE :", rmse)
print("R2 Score :", r2s)
plt.scatter(y_test,prediction)
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)])
plt.show()


X=data.drop('target',axis=1)

y=data['target']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)
prediction=model.predict(X_test)

mse=mean_squared_error(y_test,prediction)
rmse=np.sqrt(mse)

r2s=r2_score(y_test,prediction)
print("Multiple Linear Regression")
print("MSE :", mse)
print("RMSE :", rmse)
print("R2 Score :", r2s)
plt.scatter(y_test,prediction)
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)])
plt.show() 
