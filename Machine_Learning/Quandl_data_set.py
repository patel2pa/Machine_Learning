'''
used quandl to import stock data from the web, 
saved the data into panda data set, 
then used sklearn to predict the trend
'''

import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing,  svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib import style
import pickle 
style.use('ggplot')


df = pd.read_csv('https://www.quandl.com/api/v3/datasets/EOD/HD.csv?api_key=bsYZyfnBii--Dz4rC4g1')

print(df['Dividend'])
df = df[['Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume']]



df['hl_change'] = (df['Adj_High']-df['Adj_Close'])/ df['Adj_Close']*100
df['pct_change'] = (df['Adj_Close']-df['Adj_Open'])/ df['Adj_Open']*100


df = df[['Adj_Close','hl_change','pct_change', 'Adj_Volume']]

forcast_col = 'Adj_Close'

"to fill in missing data"
df.fillna(-99999, inplace = True)

"perdicting prices, for 0.1 = 10% that is the math.ceil number"
forcast_out = int(math.ceil(0.003*len(df)))


df['label'] = df[forcast_col].shift(-forcast_out)




"drop label column"
X = np.array(df.drop(['label'],1))

"scaling the data"


X = preprocessing.scale(X)
X_lately = X[-forcast_out:]
X = X[:-forcast_out]

"scaling the data"

df.dropna(inplace = True)
y = np.array(df['label'])


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2)

clf = VotingClassifier([('lsvc',svm.LinearSVC()),
                            ('knn',neighbors.KNeighborsClassifier()),
                            ('rfor',RandomForestClassifier())])
clf.fit(X_train, y_train)
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
forcast_set = clf.predict(X_lately)

k = []
for i in range(29):
    k.append(i)
print(k)

#plt.plot(k, forcast_set)
#plt.show()



