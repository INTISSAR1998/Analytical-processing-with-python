#partie 3 
from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()

x=iris.data
y=iris.target #etiquette


clf=svm.SVC(kernel='linear',C=5,decision_function_shape='ovr')
#clf=svm.OneClassSVM(kernel='rbf',gamma=0.002,nu=0.001)
#clf=svm.OneClassSVM(kernel='linear',nu=0.17)
#clf=svm.OneClassSVM(kernel='poly',nu=0.17,gamma=0.17)
#clf=svm.OneClassSVM(kernel='sigmoid',nu=0.017,gamma=0.015)

from sklearn.model_selection import train_test_split
x_train,x_test,train_label,test_label=train_test_split(x,y,random_state=0)


clf.fit(x_train,train_label)
pred=clf.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(test_label,pred))

#pas acc dans multiclass
from sklearn.metrics import accuracy_score
ACC=accuracy_score(test_label,pred)*100
print(ACC)  


#permet de calculer tf et tp =>thresholds
from sklearn.metrics import roc_curve , auc
fp,tp,thresholds=roc_curve(test_label,pred,pos_label=1)
print(fp,tp)
AUC=auc(fp,tp)*100
print(AUC)

"Autre méthode"
import pandas as pd 
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
iris = pd.read_csv('C:/Users/user/Desktop/TEK-UP/DSEN-2/Semestre 2/Traitements analytiques avec python/iris.csv')
print(iris)
iris.info()
iris.describe()
iris.columns = iris.columns.str.strip()
print(iris.columns)
x1 = iris.loc[:, "sepal.length"]
y1 = iris.loc[:, "sepal.width"]
x2 = iris.loc[:, "petal.length"]
y2 = iris.loc[:, "petal.width"]
lab = iris.loc[:, "variety"]
plt.scatter(x1[lab == 0], y1[lab == 0], x2[lab == 0], y2[lab == 0], color='blue', label='Setosa')
plt.scatter(x1[lab == 1], y1[lab == 1], x2[lab == 0], y2[lab == 0], color='red', label='Virginica')
plt.scatter(x1[lab == 2], y1[lab == 2], x2[lab == 0], y2[lab == 0], color='green', label='Versicolor')
# lable des axes
plt.xlabel('Longueur des pétales')
plt.ylabel('Largeur des pétales')
# Affichage du graphe
plt.legend()
plt.show()
import seaborn as sns
# schématiser la distribution des classes
sns.countplot(iris['variety'])


""""""""
C1= iris.loc[iris['variety'].isin(["Setosa"])]
C2= iris.loc[iris['variety'].isin(["Versicolor"])]
C3= iris.loc[iris['variety'].isin(["Virginica"])]

C1.replace("Setosa", 0, inplace=True)
C2.replace("Versicolor", 1, inplace=True)
C3.replace("Virginica", 2, inplace=True)

C1Data=C1.drop(['variety'],axis=1)
C2Data=C2.drop(['variety'],axis=1)
C3Data=C3.drop(['variety'],axis=1)

C1Target=C1['variety']
C2Target=C2['variety']
C3Target=C3['variety']

from sklearn.model_selection import train_test_split

C1_train,C1_test,C1train_label,C1test_label=train_test_split(C1Data,C1Target,test_size = 0.33 , random_state= 0)
C2_train,C2_test,C2train_label,C2test_label=train_test_split(C2Data,C2Target,test_size = 0.33 , random_state= 0)
C3_train,C3_test,C3train_label,C3test_label=train_test_split(C3Data,C3Target,test_size = 0.33 , random_state= 0)


