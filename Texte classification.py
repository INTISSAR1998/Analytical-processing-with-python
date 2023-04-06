import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

df = pd.read_csv('E:/TEK-UP/DSEN-2/Semestre 2/Traitements analytiques avec python/spam.csv')

df.info()

data = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

# Copier Data dans X et les étiquettes dans Y
X = data.v2
Y = data.v1

# Transformer les étiquettes de texte en numérique ( 0 et 1 )
le = LabelEncoder()
Y_New = le.fit_transform(Y)
Y_New =Y_New.reshape(-1,1)
Y_New.shape

# Transformer la base de test en numérique
max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X)
sequences = tok.texts_to_sequences(X)

# Mettre les données en même longueur de vecteur
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len,padding='post')
sequences_matrix

# Diviser les données en ensembles d'apprentissage et de test
from sklearn.model_selection import train_test_split
x_train1,x_test1,train_label,test_label=train_test_split(sequences_matrix,Y_New,test_size=0.33,random_state=0)




# Partie Classification


# XGBoost
import xgboost as xgb
model_Xgboost = xgb.XGBClassifier(random_state=0,n_estimators=70,max_depth=5,objectif='reg:linear').fit(x_train1,train_label)
pred_Xgboost = model_Xgboost.predict(x_test1)

# Evaluer les prédictions
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_label, pred_Xgboost)*100
print('XGBoost Accuracy : ' , accuracy)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(test_label, pred_Xgboost)
print('XGBoost Confusion matrix : ' , cm)

import matplotlib.pyplot as plt
import seaborn as sns
class_names=[0,1] 
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cm), annot = True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.title('XGBoost Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

from sklearn.metrics import classification_report
print(classification_report(test_label, pred_Xgboost))

import time
debut=time.time()
model_Xgboost.fit(x_train1,train_label)
temps_Xgboost = time.time()-debut
print('XGBoost Temps : ' , temps_Xgboost)

from sklearn.metrics import roc_curve, auc
fp, tp, thressholds=roc_curve(test_label, pred_Xgboost, pos_label=1)
print(fp,tp)
auc = auc(fp, tp)*100
print('XGBoost Area Under the Curve (AUC) : ' , auc)
    
"""roc curve"""
import matplotlib.pyplot as plt
plt.plot(fp, tp, color='blue', label='AUC = %0.2f')
plt.title('XGBoost Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(test_label, pred_Xgboost))
print('XGBoost RMSE : ' , rmse)
#print("RMSE: %f" % (rmse))

# Random Forest
from sklearn.ensemble import RandomForestClassifier
model_RandomForest = RandomForestClassifier(random_state=0,n_estimators=10).fit(x_train1,train_label)
pred_RandomForest = model_RandomForest.predict(x_test1)

from sklearn.metrics import accuracy_score
accuracy_RandomForest = accuracy_score(test_label, pred_RandomForest)*100
print('Random Forest Accuracy : ' , accuracy_RandomForest)

from sklearn.metrics import confusion_matrix
cm_RandomForest=confusion_matrix(test_label, pred_RandomForest)
print('Random Forest Confusion matrix : ' , cm_RandomForest)

import matplotlib.pyplot as plt
import seaborn as sns
class_names=[0,1] 
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cm), annot = True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.title('Random Forest Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

from sklearn.metrics import classification_report
print(classification_report(test_label, pred_RandomForest))

import time
debut=time.time()
model_RandomForest.fit(x_train1,train_label)
temps_RandomForest=time.time()-debut
print('Random Forest Temps : ' , temps_RandomForest)

from sklearn.metrics import roc_curve, auc
fp, tp, thressholds=roc_curve(test_label, pred_RandomForest, pos_label=1)
print(fp,tp)
auc_RandomForest = auc(fp, tp)*100
print('Random Forest Area Under the Curve (AUC) : ' , auc_RandomForest)
    
"""roc curve"""
import matplotlib.pyplot as plt
plt.plot(fp, tp, color='blue', label='AUC = %0.2f')
plt.title('Random Forest Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# SVM

from sklearn import svm
model_SVM=svm.SVC(kernel='rbf',C=1)
model_SVM.fit(x_train1,train_label)
pred_SVM = model_SVM.predict(x_test1)

from sklearn.metrics import accuracy_score
accuracy_SVM = accuracy_score(test_label, pred_SVM)*100
print('SVM Accuracy : ' , accuracy_SVM)

from sklearn.metrics import confusion_matrix
cm_SVM = confusion_matrix(test_label, pred_SVM)
print('SVM Confusion matrix : ' , cm_SVM)

from sklearn.metrics import classification_report
print(classification_report(test_label, pred_SVM))

import time
debut=time.time()
model_SVM.fit(x_train1,train_label)
temps_SVM = time.time()-debut
print('SVM Temps : ' , temps_SVM)
    
"""roc curve"""
import matplotlib.pyplot as plt
plt.plot(fp, tp, color='blue', label='AUC = %0.2f')
plt.title('SVM Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

dataset_selected1 = data.loc[data['Outcome'].isin([1])]
print(dataset_selected1)

dataset_selected0 = data.loc[data['Outcome'].isin([0])]
print(dataset_selected0)

label_1 = dataset_selected1['Outcome']
label_0 = dataset_selected0['Outcome']-1

data_1 = dataset_selected1.drop(['Outcome'],axis=1)
data_0 = dataset_selected0.drop(['Outcome'],axis=1)


x_train1_1,x_test1_1,train_label_1,test_label_1=train_test_split(data_1,label_1,test_size=0.33,random_state=0)

model1 = svm.OneClassSVM(nu=0.04, kernel="sigmoid", gamma=0.1)

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



# Decision Tree
from sklearn.tree import DecisionTreeClassifier
model_DecisionTree = DecisionTreeClassifier(random_state=0).fit(x_train1,train_label)
pred_DecisionTree = model_DecisionTree.predict(x_test1)

from sklearn.metrics import accuracy_score
accuracy_DecisionTree = accuracy_score(test_label, pred_DecisionTree)*100
print('Decision Tree Accuracy : ' , accuracy_DecisionTree)

from sklearn.metrics import confusion_matrix
cm_DecisionTree = confusion_matrix(test_label, pred_DecisionTree)
print('Decision Tree Confusion matrix : ' , cm_DecisionTree)

import matplotlib.pyplot as plt
import seaborn as sns
class_names=[0,1] 
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cm), annot = True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.title('Decision Tree Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

from sklearn.metrics import classification_report
print(classification_report(test_label, pred_DecisionTree))

import time
debut=time.time()
model_DecisionTree.fit(x_train1,train_label)
temps_DecisionTree = time.time()-debut
print('Decision Tree Temps : ' , temps_DecisionTree)

from sklearn.metrics import roc_curve, auc
fp, tp, thressholds=roc_curve(test_label, pred_Xgboost, pos_label=1)
print(fp,tp)
auc_DecisionTree = auc(fp, tp)*100
print('Decision Tree Area Under the Curve (AUC) : ' , auc_DecisionTree)
    
"""roc curve"""
import matplotlib.pyplot as plt
plt.plot(fp, tp, color='blue', label='AUC = %0.2f')
plt.title('Decision Tree Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Pour svm binaire on essaye kernel rbf et pour svm monoclass on essaye kernel sigmoid