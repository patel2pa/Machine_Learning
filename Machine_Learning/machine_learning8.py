#https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import pandas as pd
import random 

'''
#example of data analysis using KMean method 
x = np.array([[1,2], [1.5,1.8], [5,8],[1,0.6],[7,8], [6,9]])
#plt.scatter(x[:,0],x[:,1])

#plt.show()

clf = KMeans(n_clusters = 2)

clf.fit(x)

centroids = clf.cluster_centers_
labels = clf.labels_

for i in range(len(x)):
    plt.plot(x[i][0],x[i][1])
plt.scatter(centroids[:,0], centroids[:,1])
plt.show()
print(centroids)
'''



df = pd.read_csv('titanic.csv')
orginal_df = pd.read_csv('titanic.csv')
df.drop(['body','name'],1, inplace=True)
#df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)


def handle_non_numarical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int,df[column]))

    return df

df = handle_non_numarical_data(df)





x = np.array(df.drop(['survived'],1).astype(float))
x = preprocessing.scale(x)
y = np.array(df['survived'])

print(x)
clf = KMeans(n_clusters = 2)
clf.fit(x)

co = 0
for i in range(len(x)):
    this = x[i].reshape(-1,len(x[i]))
    prediction = clf.predict(this)
    if prediction[0]==y[i]:
        co = co + 1
#print(co)
#print(co/len(x))
#print(prediction)

centroids = clf.cluster_centers_


labels = clf.labels_ 
    
''' 
correct = 0
for i in range(len(x)):
    #predict_me = np.array(x[i].astype(float))
    #predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0]== y[i]:
        correct +=1
    print(prediction) 
'''



orginal_df['cluster_group'] = np.nan

for i in range(len(x)):
    orginal_df['cluster_group'].iloc[i] = labels[i]
#assining labels to the original group



n_clusters_ = len(np.unique(labels))

  
survival_rates = {}
for i in range(n_clusters_):
    temp_df = orginal_df[(orginal_df['cluster_group']==float(i))]
    survival_cluster = temp_df[(temp_df['survived']==1)]
    survival_rate = len(survival_cluster) / len(temp_df)
    survival_rates[i]= survival_rate
    print(orginal_df[(orginal_df['cluster_group']==i)].describe())

print(survival_rates)   










'''
# tryed to do this with a class but no good
x = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]])
print(x)

class K_Means:
    
    def __init__(self, k=2, tol = 0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids= {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifictions[i] = []

            for futureset in x:
                distances = [np.linalg.norm()]
                

    def predict(self,data):
        pass
'''


'''
df = pd.read_csv('titanic.csv')
# defining df to be the value of the titanic.csv file
df.drop(['body','name'],1, inplace=True)
#df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)
#filling in missing data

# the following data is converting data that is not numarical in to int
def handle_non_numarical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int,df[column]))

    return df

# the end result of the conversion
df = handle_non_numarical_data(df)





x = np.array(df.drop(['survived'],1).astype(float))
x = preprocessing.scale(x)
y = np.array(df['survived'])
'''


'''
Following is converted data, which is put thru a k.mean algo and the end
result is 2 cluster's centroid(two different centroids).
The first step is to two pick random points, then find the closes set of two data points.
Then find the avarge of all those point, repeat unit the centroid stops moving.
'''

'''
z = 0

var_1 = x[random.randint(0,len(x)-1)]
var_2 = x[random.randint(0,len(x)-1)]
while z <100: 
    
    

    var_1_list = []
    var_2_list = []




    for num in x:
        dis = 0
        dis_2 = 0
        for var in range(len(num)):
            dis = dis + ((var_1[var]) -  (num[var]))**2
            dis_2 = dis_2 + ((var_2[var]) -  (num[var]))**2
            
        if dis_2>dis:
            var_1_list.append(num)
            
        if dis_2<dis:
            var_2_list.append(num)
        dis = 0
        dis_2 = 0

    var_1 = (np.mean(var_1_list,axis=0))
    var_2 = (np.mean(var_2_list,axis=0))

    z  = z + 1

print(var_1)
print(var_2)

'''    
        
   





    




        
            
            
            
