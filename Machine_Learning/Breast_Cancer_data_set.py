'''
Used State vector machine from the Sklearn libery to 
analyze chanaces of breast cancer with given parameters. 
'''
import pandas as pd
import numpy as np
from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

df = pd.read_csv('breast_can_data.txt')


"filling in the gap, -9999 is for outliers"
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)


X = np.array(df.drop(['classes'],1))
y = np.array(df['classes'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)


accuracy = clf.score(X_test, y_test)


example_label = np.array([4,2,1,1,1,2,3,2,1])

'reshape is to make it into the right formate for sklearn'
example_label = example_label.reshape(1,-1) 
print(example_label)
prediction = clf.predict(example_label)
print(prediction)
print(accuracy)
