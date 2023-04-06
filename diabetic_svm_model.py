# -*- coding: utf-8 -*-
"""diabetic-svm-model.ipynb

# This solution uses SVM a supervised machine learning model using monoclass classification with one versus all and one versus one treating the case of inbalanced data each step will be explained with details bellow

# In this part we will import libraries needed for calulating accuracy , splitting data , importing svm model from sklearn 
# -we will feed our model with the dataset in csv file 
# -reading our data set 
# - reorganizing the data set to be more readble
"""

#Data Pre-processing Step  
# importing libraries  
import numpy as np 
import pandas as pd 
from sklearn import svm 
from sklearn.model_selection import train_test_split 
import time 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score   
#charging the dataset 
print('----------------------- feeding data  ----------------------------------')
ds =pd.read_csv('E:/TEK-UP/DSEN-2/Semestre 2/Traitements analytiques avec python/diabetes.csv')
print (ds)
#dataset description
ds.info()
ds.describe()
label = ds['Outcome']
Data = ds.drop(['Outcome'], axis=1 )

"""# In this part we will split the data into training data and validation data"""

#splitting the datat into training data and test data
x_train1,x_test1, train_label,test_label =train_test_split(Data, label, test_size=0.33 , random_state=0)

"""# In this part we will use SVM with linear kernel ,RBF , Polynomial and sigmoid and in each kernel we will 
# -calculate the accuracy 
# -calculate the recall AUC 
# -calculate the CC
# -visualize the confusion matrix
# -visualize the roc curve plot 
# -Balancing the data while treating the model like multiclass

Here we will classify the data with
"""

#support vector machine 
#****************************   Linear kernel   ***********************************************************#
model =svm.SVC(kernel='linear',C=1)
model.fit(x_train1,train_label)
print("------------------- prediction part for Linear kernel -----------------------")
pred=model.predict(x_test1)
print(pred)
#calculate the accuracy 
ACC=accuracy_score(test_label,pred)*100
print("--------------------  Accuracy score for Linear kernel ---------------------")
print(ACC)
print ('---------------------  FIRST CONFUSION MATRIX for Linear kernel-------------------------------')
results = confusion_matrix(test_label, pred)
print(results)
#report 
print ("---------------------  Linear kernel  classification report  -------------------------- ")
print(classification_report(test_label,pred))

debut =time.time()
model.fit(x_train1,train_label)
Temps=time.time()-debut

print("----------------  Linear kernel  time training calcul --------------------- ")
print(Temps)
#Recall test 
print ("/***********Linear kernel  Recall calcul *****/")
fp, tp , thresholds =roc_curve(test_label, pred, pos_label=1)
print(fp, tp)
AUC=auc(fp,tp)*100
print('Recall: %.3f' % AUC)
#roc curve 
print ("/****** VISUALIZATION OF ROC CURVE for Linear kernel  ********/")
plt.plot(fp, tp, color='blue',label = 'AUC = %0.2f' % AUC)
plt.title('Receiver Operating Characteristic')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % AUC)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#print with heatmap

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
# CM1=pd.DataFrame(CM)
# print(CM1)
sns.heatmap(pd.DataFrame(results), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
#plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


#****************************   RBF kernel   ***********************************************************#
print("/*** Balance data for RBF kernel *******/")
dataset_selected1= ds.loc[ds ['Outcome'].isin([1])]
dataset_selected0= ds.loc[ds ['Outcome'].isin([0])]
data_1=dataset_selected1.drop(['Outcome'],axis=1)
data_0=dataset_selected0.drop(['Outcome'],axis=1)
label_1=dataset_selected1['Outcome']
label_0=dataset_selected0['Outcome']-1
#splitting data
x_train1,x_test1,train_label,test_label=train_test_split(data_1,label_1,test_size=0.33,random_state=0)
#apply rbf kernel in prediction
model1=svm.OneClassSVM(kernel='rbf',gamma=0.1,nu=0.1)
test_data=np.concatenate((x_test1,data_0),axis=0)
test_label_total=np.concatenate((test_label,label_0),axis=0)
model1.fit(x_train1)
pred1=model1.predict(test_data)

#Accuracy
print ("/***** RBF accuraccy value ***/")
ACC1=accuracy_score(test_label_total,pred1)*100
print(ACC1)

# 
print ('/*********   RBF CONFUSION MATRIX  *****/')

from sklearn.metrics import confusion_matrix
M1=confusion_matrix(test_label_total,pred1)
print("/******second confusion matrix ********/")
print (M1)
#report 
print ("/************** classification report RBF kernel  **************/")
from sklearn.metrics import classification_report
print(classification_report(test_label_total,pred1))
#time CC
import time 
debut =time.time()
model.fit(test_data, test_label_total)
Temps1=time.time()-debut

print("/*********** time training calcul RBF kernel  ***************/")
print(Temps1)
#Recall test 
print ("/*********** Recall calcul RBF kernel  *****/")
from sklearn.metrics import roc_curve, auc 
fp1, tp1 , thresholds =roc_curve(test_label_total, pred1, pos_label=1)
print(fp1, tp1)
AUC_rbf=auc(fp1,tp1)*100
print('Recall: %.3f' % AUC_rbf)
#roc curve 
print ("/****** VISUALIZATION OF ROC CURVE for RBF kernel ********/")
import matplotlib.pyplot as plt
plt.plot(fp1, tp1, color='blue',label = 'AUC = %0.2f' % AUC_rbf)
plt.title('Receiver Operating Characteristic')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % AUC)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#****************************   polyminal kernel   ***********************************************************#

print ("/************  polyminal kernel  ********/ ")
dataset_selected1= ds.loc[ds ['Outcome'].isin([1])]
dataset_selected0= ds.loc[ds ['Outcome'].isin([0])]
data_1=dataset_selected1.drop(['Outcome'],axis=1)
data_0=dataset_selected0.drop(['Outcome'],axis=1)
label_1=dataset_selected1['Outcome']
label_0=dataset_selected0['Outcome']-1
#splitting data
from sklearn.model_selection import train_test_split
x_train1,x_test1,train_label,test_label=train_test_split(data_1,label_1,test_size=0.33,random_state=0)
#apply poly kernel in prediction
model1=svm.OneClassSVM(kernel='poly',gamma=0.1,nu=0.1)
test_data=np.concatenate((x_test1,data_0),axis=0)
test_label_total=np.concatenate((test_label,label_0),axis=0)
model1.fit(x_train1)
pred1=model1.predict(test_data)

#Accuracy
print ("/***** accuraccy value ***/")
from sklearn.metrics import accuracy_score
ACC1=accuracy_score(test_label_total,pred1)*100
print(ACC1)

# 
print ('/*********   polynomial CONFUSION MATRIX  *****/')

from sklearn.metrics import confusion_matrix
M1=confusion_matrix(test_label_total,pred1)
print("/****** polynomial confusion matrix ********/")
print (M1)
#report 
print ("/************** classification report polynomial kernel  **************/")
from sklearn.metrics import classification_report
print(classification_report(test_label_total,pred1))
#time CC
import time 
debut =time.time()
model.fit(test_data, test_label_total)
Temps1=time.time()-debut

print("/*********** time training calcul polynomial kernel ***************/")
print(Temps1)
#Recall test 
print ("/*********** Recall calcul polynomial kernel   *****/")
from sklearn.metrics import roc_curve, auc 
fp2, tp2 , thresholds =roc_curve(test_label_total, pred1, pos_label=1)
print(fp2, tp2)
AUC_poly=auc(fp,tp)*100
print('Recall: %.3f' % AUC_poly)
#roc curve 
print ("/****** VISUALIZATION OF ROC CURVE for polynomial kernel ********/")
import matplotlib.pyplot as plt
plt.plot(fp2, tp2, color='blue',label = 'AUC = %0.2f' % AUC_poly)
plt.title('Receiver Operating Characteristic')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % AUC)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#****************************   sigmoid kernel   ***********************************************************#

print ("/************  Sigmoid kernel  ********/ ")
#*******Sigmoid kernel ********
print("/*** part of Sigmoid kernel *******/")
dataset_selected1= ds.loc[ds ['Outcome'].isin([1])]
dataset_selected0= ds.loc[ds ['Outcome'].isin([0])]
data_1=dataset_selected1.drop(['Outcome'],axis=1)
data_0=dataset_selected0.drop(['Outcome'],axis=1)
label_1=dataset_selected1['Outcome']
label_0=dataset_selected0['Outcome']-1
#splitting data
from sklearn.model_selection import train_test_split
x_train1,x_test1,train_label,test_label=train_test_split(data_1,label_1,test_size=0.33,random_state=0)
#apply Sigmoid kernel in prediction
model1=svm.OneClassSVM(kernel='sigmoid',gamma=0.1,nu=0.1)
test_data=np.concatenate((x_test1,data_0),axis=0)
test_label_total=np.concatenate((test_label,label_0),axis=0)
model1.fit(x_train1)
pred1=model1.predict(test_data)

#Accuracy
print ("------------------- Sigmoid accuraccy value --------------------------")
from sklearn.metrics import accuracy_score
ACC1=accuracy_score(test_label_total,pred1)*100
print(ACC1)

# 
print ('/*********   Sigmoid  CONFUSION MATRIX ------------------------')

from sklearn.metrics import confusion_matrix
M1=confusion_matrix(test_label_total,pred1)
print("-------------------------- Sigmoid  confusion matrix---------------------")
print (M1)
#report 
print ("------------------ classification report Sigmoid  kernel  -------------------------------")
from sklearn.metrics import classification_report
print(classification_report(test_label_total,pred1))
#time CC Sigmoid
import time 
debut =time.time()
model.fit(test_data, test_label_total)
Temps1=time.time()-debut

print("--------------------- time training calcul Sigmoid kernel ----------------------------")
print(Temps1)
#Recall test Sigmoid
print ("--------------------- Recall calcul Sigmoid kernel -----------------------------------------")
from sklearn.metrics import roc_curve, auc 
fp3, tp3 , thresholds =roc_curve(test_label_total, pred1, pos_label=1)
print(fp3, tp3)
AUC_sig=auc(fp3,tp3)*100
print('Recall: %.3f' % AUC_sig)
#roc curve Sigmoid
print ("------------------- VISUALIZATION OF ROC CURVE for Sigmoid kernel ---------------------")
import matplotlib.pyplot as plt
plt.plot(fp3, tp3, color='blue',label = 'AUC = %0.2f' % AUC_sig)
plt.title('Receiver Operating Characteristic')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % AUC)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

"""# This part visualize the four kernels in one plot """

print("-------------------------- 4 kernel ROC curve --------------------")

plt.title('Receiver Operating Characteristic')
plt.plot(fp, tp, color='blue',label = 'AUC Linear = %0.2f' % AUC)
plt.plot(fp1, tp1, color='red',label = 'AUC RBF = %0.2f' % AUC_rbf)
plt.plot(fp2, tp2, color='yellow',label = 'AUC Poly = %0.2f' % AUC_poly)
plt.plot(fp3, tp3, color='black',label = 'AUC Sig = %0.2f' % AUC_sig)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()