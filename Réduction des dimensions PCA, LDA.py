# load and show an image with Pillow
from PIL import Image

import numpy as np

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#Partie I : Préparation des données
# Open the image form working directory
img=Image.open('E:/TEK-UP/DSEN-2/Semestre 2/Traitements analytiques avec python/yalefaces/subject01.happy')

import matplotlib.pyplot as plt
# display the array of pixels as an image
plt.imshow(img,'gray')

img1 = np.array(img)
img2 = img1.reshape(img1.shape[0]*img1.shape[1])

import glob

path = glob.glob('E:/TEK-UP/DSEN-2/Semestre 2/Traitements analytiques avec python/yalefaces/*')

vector=[]
for name in path:
    print (name)
    im = Image.open(name)
    imArray = np.array(im)
    im1 = imArray.reshape(imArray.shape[0]*imArray.shape[1])
    vector.append(im1)
    
 
#Partie II : Réduction de dimension en utilisant PCA et classification par SVM
from sklearn.decomposition import PCA # PCA

#transform vector to an array
arr=np.asarray(vector)
    
#apply transform to the array
#pca = PCA(n_components=165).fit_transform(arr)
#pca = PCA(n_components=100).fit_transform(arr)
#pca = PCA(n_components=50).fit_transform(arr)
#pca = PCA(n_components=25).fit_transform(arr)
pca = PCA(n_components=15).fit_transform(arr)

#Build the vector of labels
label=[]
label=np.repeat(range(1,16),11)

#Split the database into 2/3 for training and 1/3 for testing
from sklearn.model_selection import train_test_split
x_train,x_test,train_label,test_label=train_test_split(pca,label,test_size=0.33,random_state=0)




#support vector machine
from sklearn import svm
model=svm.SVC(kernel='linear',C=1)
model.fit(x_train,train_label)#train/apprentissage
pred=model.predict(x_test)#test

#Evaluate the performance of the classifier in terms of ACC
linear_ACC=accuracy_score(test_label, pred)*100
print(linear_ACC)


#Evaluate the performance of the classifier in terms of Confusion Matrix
import pandas as pd 

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(test_label,pred)
print(cm)

#Visualizing Confusion Matrix using Heatmap
import matplotlib.pyplot as plt
import seaborn as sns
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#Partie III : Réduction de dimension en utilisant et classification LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

model = LDA()
X2 = model.fit_transform(arr,label)

X2_train,X2_test,Y2_train,Y2_test=train_test_split(X2,label,test_size=0.33,random_state=0)

model.fit(X2_train,Y2_train)

predicted2 = model.predict(X2_test)

from sklearn.metrics import accuracy_score
print( "LDA accuracy" + str(100 * accuracy_score(Y2_test, predicted2)))