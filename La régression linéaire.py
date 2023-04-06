# import pandas as pd 
# from sklearn import svm
# import numpy as np
# dataset = pd.read_csv('C:/Users/user/Desktop/TEK-UP/DSEN-2/Semestre 2/Traitements analytiques avec python/house-prices.csv')
# print(dataset)
# dataset.info()
# dataset.describe()

# size = np.array(dataset["SqFt"])
# Real_regression = np.array(dataset["Price"])

# X1 = size.reshape(-1,1)
# Y1 = Real_regression.reshape(-1,1)


# import matplotlib.pyplot as plt
# #plt.plot(X1,Y1, 'r*')
# plt.scatter(X1,Y1)
# plt.xlabel('Axe des absciesse')
# plt.ylabel('Axe des ordonnées')

# from sklearn.linear_model import LinearRegression

# model = LinearRegression()
# model.fit(X1,Y1) #calcul de a et b les parametres de notre modèle

# a = model.coef_
# b = model.intercept_

# label = model.predict(X1)
# Label1 = a*X1+b

# from sklearn.metrics import mean_squared_error
# import math

# mse = mean_squared_error(Y1, label)

# rmse = math.sqrt(mse)

# print('RMSE', rmse)

# import pandas as pd
# Y1 = pd.DataFrame(Y1)
# Y1.describe()

# from sklearn.metrics import explained_variance_score
# EV = explained_variance_score(Y1, label)
# print("Explained variance: %f" % (EV))


# import matplotlib.pyplot as plt
# plt.scatter(X1,Y1,color="green") #real
# plt.plot(X1,label,"blue") #predict

# plt.title('Linear Regression')
# plt.xlabel('Size')
# plt.ylabel('Price')

#########################################################################
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

from sklearn.linear_model import LinearRegression

model1 = LinearRegression().fit(X1,Y) 


a = model1.coef_
b = model1.intercept_

label = model1.predict(X1)

from sklearn.metrics import mean_squared_error
import math

mse = mean_squared_error(Y, label)

rmse = math.sqrt(mse)

print('RMSE régression linéaire multiple', rmse)

from sklearn.metrics import explained_variance_score
EV = explained_variance_score(Y, label)
print("Explained variance régression linéaire multiple: %f" % (EV))


# Complexité de calcul de régression linéaire multiple
import time
debut=time.time()
model1 = LinearRegression().fit(X1,Y) 
Temps=time.time()-debut
print(Temps)

