import pandas as pd 
from sklearn import svm
import numpy as np
dataset = pd.read_csv('C:/Users/user/Desktop/TEK-UP/DSEN-2/Semestre 2/Traitements analytiques avec python/iris.csv')
print(dataset)
dataset.info()
dataset.describe()

Data = dataset.drop(['variety'], axis=1)
Label = dataset.loc[:, "variety"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Data, Label, train_size=0.66, random_state=0)

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

model.fit(X_train,y_train)#train/apprentissage
pred=model.predict(X_test)#test

from sklearn.metrics import accuracy_score
ACC=accuracy_score(y_test, pred)*100
print(ACC)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,pred)
print(cm)

"""Visualizing Confusion Matrix using Heatmap"""
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
fig, ax = plt.subplots()
ax.xaxis.set_label_position("top")
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
tree.plot_tree(model, filled=True)