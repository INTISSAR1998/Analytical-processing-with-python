import pandas as pd 
from sklearn import svm
import numpy as np
dataset = pd.read_csv('C:/Users/user/Desktop/TEK-UP/DSEN-2/Semestre 2/Traitements analytiques avec python/house-prices.csv')
print(dataset)

price = dataset['Price']
Y = np.array(price).reshape(-1,1)

Data = dataset.drop(['Price','Brick','Neighborhood','Home'],axis=1)

from sklearn.preprocessing import LabelEncoder
X2 = dataset['Brick']
le = LabelEncoder()
X2New = le.fit_transform(X2)
X2New = X2New.reshape(-1,1)

from sklearn.preprocessing import LabelEncoder
X3 = dataset['Neighborhood']
le = LabelEncoder()
X3New = le.fit_transform(X3)
X3New = X3New.reshape(-1,1)

X1 = np.concatenate((Data, X2New, X3New), axis=1)

from sklearn.ensemble import RandomForestRegressor
model1 = RandomForestRegressor(random_state=0).fit(X1,Y) 


a = model1.coef_
b = model1.intercept_

label = model1.predict(X1)

from sklearn.metrics import mean_squared_error
import math

mse = mean_squared_error(Y, label)

rmse = math.sqrt(mse)

print('RMSE régression non linéaire', rmse)

from sklearn.metrics import explained_variance_score
EV = explained_variance_score(Y, label)
print("Explained variance régression non linéaire: %f" % (EV))


# Complexité de calcul de régression non linéaire
import time
debut=time.time()
model1 = RandomForestRegressor(random_state=0).fit(X1,Y) 
Temps=time.time()-debut
print(Temps)

