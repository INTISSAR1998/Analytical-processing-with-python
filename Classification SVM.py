'''classification binaire'''
import numpy as np
import pandas as pd 
from sklearn import svm
dataset = pd.read_csv('C:/Users/user/Desktop/TEK-UP/DSEN-2/Semestre 2/Traitements analytiques avec python/diabetes.csv')
print(dataset)
dataset.info()
dataset.describe()
label=dataset['Outcome']
"axis=1 colonne, axis=0 ligne"
data = dataset.drop(['Outcome'],axis=1)
'classification'
'deviser base en 2/3 et 1/3'
from sklearn.model_selection import train_test_split
x_train1,x_test1,train_label,test_label=train_test_split(data,label,test_size=0.33,random_state=0)


'support vector machine'
model=svm.SVC(kernel='linear',C=1)
model.fit(x_train1,train_label)#train/apprentissage
pred=model.predict(x_test1)#test

from sklearn.metrics import accuracy_score
ACC=accuracy_score(test_label, pred)*100
print(ACC)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(test_label,pred)
print(cm)


"""Visualizing Confusion Matrix using Heatmap"""
import matplotlib.pyplot as plt
import seaborn as sns
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
# CM1=pd.DataFrame(CM)
# print(CM1)
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
#plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


from sklearn.metrics import classification_report
print(classification_report(test_label, pred))

import time
debut=time.time()
model.fit(x_train1,train_label)
Temps=time.time()-debut
print(Temps)

from sklearn.metrics import roc_curve, auc
fp, tp, thresholds = roc_curve(test_label, pred, pos_label=1)
print(fp,tp)
AUC=auc(fp, tp)*100
print(AUC)

"""roc curve"""
import matplotlib.pyplot as plt
plt.plot(fp, tp, color='blue', label='AUC = %0.2f')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

label.value_counts()

dataset_selected1 = dataset.loc[dataset['Outcome'].isin([1])]
print(dataset_selected1)

dataset_selected0 = dataset.loc[dataset['Outcome'].isin([0])]
print(dataset_selected0)

label_1 = dataset_selected1['Outcome']
label_0 = dataset_selected0['Outcome']-1

data_1 = dataset_selected1.drop(['Outcome'],axis=1)
data_0 = dataset_selected0.drop(['Outcome'],axis=1)


x_train1_1,x_test1_1,train_label_1,test_label_1=train_test_split(data_1,label_1,test_size=0.33,random_state=0)

model1 = svm.OneClassSVM(nu=0.04, kernel="rbf", gamma=0.1)

model1.fit(x_train1_1)


import numpy as np
data_test= np.concatenate((x_test1_1, data_0),axis=0)

label_test = np.concatenate((test_label_1, label_0),axis=0)


new_pred=model1.predict(data_test)#test

new_ACC=accuracy_score(label_test, new_pred)*100
print(new_ACC)

new_fp, new_tp, thresholds = roc_curve(label_test, new_pred, pos_label=1)
print(new_fp,new_tp)
new_AUC=auc(new_fp, new_tp)*100
print(new_AUC)


"""roc curve"""
import matplotlib.pyplot as plt
plt.plot(new_fp, new_tp, color='blue', label='new_AUC = %0.2f')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

from sklearn.metrics import classification_report
print(classification_report(label_test, new_pred))

import time
rbf_debut=time.time()
model1.fit(x_train1_1,train_label_1)
rbf_Temps=time.time()-rbf_debut
print(rbf_Temps)

""" ..."""
model2 = svm.OneClassSVM(coef0=1, kernel="sigmoid", gamma=0.1)

model2.fit(x_train1_1)


import numpy as np
data_test= np.concatenate((x_test1_1, data_0),axis=0)

label_test = np.concatenate((test_label_1, label_0),axis=0)


sigmoid_pred=model2.predict(data_test)#test

sigmoid_ACC=accuracy_score(label_test, sigmoid_pred)*100
print(sigmoid_ACC)

sigmoid_fp, sigmoid_tp, thresholds = roc_curve(label_test, sigmoid_pred, pos_label=1)
print(sigmoid_fp,sigmoid_tp)
sigmoid_AUC=auc(sigmoid_fp, sigmoid_tp)*100
print(sigmoid_AUC)


"""roc curve"""
import matplotlib.pyplot as plt
plt.plot(sigmoid_fp, sigmoid_tp, color='blue', label='sigmoid_AUC = %0.2f')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

from sklearn.metrics import classification_report
print(classification_report(label_test, new_pred))

from sklearn.metrics import confusion_matrix
sigmoid_cm=confusion_matrix(test_label,pred)
print(sigmoid_cm)
""" ..."""
model3 = svm.OneClassSVM(nu=0.1, kernel="linear")

model3.fit(x_train1_1)


import numpy as np
data_test= np.concatenate((x_test1_1, data_0),axis=0)

label_test = np.concatenate((test_label_1, label_0),axis=0)


linear_pred=model3.predict(data_test)#test

linear_ACC=accuracy_score(label_test, linear_pred)*100
print(linear_ACC)

linear_fp, linear_tp, thresholds = roc_curve(label_test, linear_pred, pos_label=1)
print(linear_fp,linear_tp)
linear_AUC=auc(linear_fp, linear_tp)*100
print(linear_AUC)


"""roc curve"""
import matplotlib.pyplot as plt
plt.plot(linear_fp, linear_tp, color='blue', label='linear_AUC = %0.2f')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

from sklearn.metrics import classification_report
print(classification_report(label_test, linear_pred))

from sklearn.metrics import confusion_matrix
linear_cm=confusion_matrix(test_label,pred)
print(linear_cm)
""""""


import pandas as pd 
from sklearn import svm
import numpy as np
from sklearn import svm
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
iris =  pd.read_csv('C:/Users/user/Desktop/TEK-UP/DSEN-2/Semestre 2/Machine learning/iris.csv')
print(iris)
"separate features set X from the target column (class label) y, and divide the data set to 80% for training, and 20% for testing:"
X = iris.data[:, :2]
y = iris.target
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=101)
"We’ll create two objects from SVM, to create two different classifiers; one with Polynomial kernel, and another one with RBF kernel"
rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)
"To calculate the efficiency of the two models, we’ll test the two classifiers using the test data set:"
poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)
"Finally, we’ll calculate the accuracy and f1 scores for SVM with Polynomial kernel:"
poly_accuracy = accuracy_score(y_test, poly_pred)
poly_f1 = f1_score(y_test, poly_pred, average='weighted')
print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))
"In the same way, the accuracy and f1 scores for SVM with RBF kernel"
rbf_accuracy = accuracy_score(y_test, rbf_pred)
rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))
"""Accuracy (Polynomial Kernel): 70.00
F1 (Polynomial Kernel): 69.67
Accuracy (RBF Kernel): 76.67
F1 (RBF Kernel): 76.36"""










#Importing the necessary packages and libaries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
iris = datasets.load_iris()
#Store variables as target y and the first two features as X (sepal length and sepal width of the iris flowers)
X = iris.data[:, :2]
y = iris.target
linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X_train, y_train)
sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(X_train, y_train)
#stepsize in the mesh, it alters the accuracy of the plotprint
#to better understand it, just play with the value, change it and print it
h = .01
#create the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
# create the title that will be shown on the plot
titles = ['Linear kernel','RBF kernel','Polynomial kernel','Sigmoid kernel']
for i, clf in enumerate((linear, rbf, poly, sig)):
#defines how many plots: 2 rows, 2columns=> leading to 4 plots
plt.subplot(2, 2, i + 1) #i+1 is the index
#space between plots
plt.subplots_adjust(wspace=0.4, hspace=0.4)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.PuBuGn, alpha=0.7)
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.PuBuGn,     edgecolors='grey')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title(titles[i])
plt.show()
linear_pred = linear.predict(X_test)
poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)
sig_pred = sig.predict(X_test)
# retrieve the accuracy and print it for all 4 kernel functions
accuracy_lin = linear.score(X_test, y_test)
accuracy_poly = poly.score(X_test, y_test)
accuracy_rbf = rbf.score(X_test, y_test)
accuracy_sig = sig.score(X_test, y_test)
print(“Accuracy Linear Kernel:”, accuracy_lin)
print(“Accuracy Polynomial Kernel:”, accuracy_poly)
print(“Accuracy Radial Basis Kernel:”, accuracy_rbf)
print(“Accuracy Sigmoid Kernel:”, accuracy_sig
# creating a confusion matrix
cm_lin = confusion_matrix(y_test, linear_pred)
cm_poly = confusion_matrix(y_test, poly_pred)
cm_rbf = confusion_matrix(y_test, rbf_pred)
cm_sig = confusion_matrix(y_test, sig_pred)
print(cm_lin)
print(cm_poly)
print(cm_rbf)
print(cm_sig)