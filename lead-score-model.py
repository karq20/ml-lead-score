#data preprocessing

#importing the datasets

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing the dataset

dataset = pd.read_csv('data_25.csv')
dataset["lead type"] = dataset["lead type"].astype(str)

print(dataset)

#creating the independent matrix
x = dataset.iloc[:, :-1].values
print(x)
#creating the dependent vector

y = dataset.iloc[:,8].values
print(y)


#taking care of missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN',strategy = 'mean',axis = 0)
#imputer = imputer.fit(x[:,1:3])
#x[:,1:3] = imputer.transform(x[:,1:3])
#print(x)

#encoding categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
#labelencoder_x.fit_transform(x[:,4])
#labelencoder_x.fit_transform(x[:,1])
#labelencoder_x.fit_transform(x[:,2])
#labelencoder_x.fit_transform(x[:,3])
#labelencoder_x.fit_transform(x[:,6])
x[:,4] = labelencoder_x.fit_transform(x[:,4])
x[:,1] = labelencoder_x.fit_transform(x[:,1])
x[:,2] = labelencoder_x.fit_transform(x[:,2])
x[:,3] = labelencoder_x.fit_transform(x[:,3])
x[:,6] = labelencoder_x.fit_transform(x[:,6])
print(x)
onehotencoder = OneHotEncoder(categorical_features = [1,4,2,3,6])
x= onehotencoder.fit_transform(x).toarray()
print(x)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#splitting the data into test and training modules
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

#feature scaling for normalization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
print(x_train)


#applying PCA
#from sklearn.decomposition import PCA
#pca = PCA(n_components = 2)
#x_train = pca.fit_transform(x_train)
#x_test = pca.fit_transform(x_test)
#explained_variance =  pca.explained_variance_ratio_

#running logistic regression - 73%
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state = 0)
#classifier.fit(x_train,y_train)

#setting up the right classifier for work - DECISION TREE - 80%
#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
#classifier.fit(x_train,y_train)


#setting up the right classifier for work - random forest - 81% 
#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators = 1000,criterion = 'entropy',random_state =0)
#classifier.fit(x_train,y_train)

#fitting the naive bayes classifier - didnt classify
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#classifier.fit(x_train,y_train)

#fitting the kernel SVM Classifier
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',random_state = 0)
classifier.fit(x_train,y_train)


#predicting the test set results 
y_pred = classifier.predict(x_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

#visualizaing the test set results
#from matplotlib.colors import ListedColormap
#x_set,y_set = x_test,y_test
#x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min() -1 , stop = x_set[:,0].max() + 1,step = 0.01),np.arange(start = x_set[:,1].min() -1 , stop = x_set[:,1].max() + 1,step = 0.01))
#plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha = 0.75,cmap = ListedColormap(('red','green')))
#plt.xlim(x1.min(),x1.max())
#plt.ylim(x2.min(),x2.max())
#for i,j in enumerate(np.unique(y_set)):
#	plt.scatter(x_set[y_set == j,0],x_set[y_set == j,1],c = ListedColormap(('red','green'))(i),label = j)
#plt.title('Decision tree classifier(training set)')
#plt.xlabel('lead ')
#plt.ylabel('conversion')
#plt.show()


#print  the probabilities 

prediction =  classifier.predict(x_test)
print(prediction)

#probabilities
probs = classifier.predict_proba(prediction)

#print the predicted gender
print(prediction)
print(probs)





















