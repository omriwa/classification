# import dataset
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values
# import regression model
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)

#  Test
x_test = [[6.5],[7.8]]
y_pred = regressor.predict(x_test)

# Visualsing
import  matplotlib.pyplot as plt

plt.scatter(X,Y, color="green")
plt.scatter(x_test,y_pred, color="red")
plt.plot(X,regressor.predict(X),color="blue")
plt.title("salary vs level")
plt.xlabel("level")
plt.ylabel("salary")
plt.show()

from  sklearn.preprocessing import PolynomialFeatures
import numpy as np

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
poly_scaler = PolynomialFeatures(degree=5)
plt.scatter(X,Y, color="green")
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title("salary vs level 2")
plt.xlabel("level")
plt.xlabel("salary")
plt.show() 