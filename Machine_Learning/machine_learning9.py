'''
probability of two independet event happing is given by
p(A) and p(B): p(A)*p(B) 

In statistics, if we have two events (A and B),
we write the probability that event A will happen,
given that event B already happened as P(A|B).
In our example, we want to find P(rare disease | positive result).
In other words, we want to find the probability
that the patient has the disease given the test came back positive.


'''


import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import pandas as pd
import random

'''
df = pd.read_csv('titanic.csv')
orginal_df =  pd.DataFrame.copy(df)
#to coopy the dara frame 
print(df.columns)
print(df.survived)

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


clf = MeanShift()
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
labels = clf.labels_
# labels for the cluster
centroids = clf.cluster_centers_




#print(centroids)
#print(len(centroids))



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
    detailed = orginal_df[(orginal_df['cluster_group']==i)].describe()
    more_detailed = detailed[(detailed['pclass']==1)]
    print(more_detailed.describe())
print(survival_rates)    

'''


X = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11],[8,2],[10,2], [9,3]])


class Mean_Shift:
    def __init__(self, radius=6):
        self.radius = radius

    def fit(self, data):
        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]


        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    if np.linalg.norm(featureset - centroid) < self.radius:
                        in_bandwidth.append(featureset)

                new_centroid = np.average(in_bandwidth, axis = 0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))

            prev_centroids = dict(centroids)

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False

                if not optimized:
                    break

            if not optimized:
                break

            if optimized:
                break

        self.centroids = centroids

    def predict(self,data):
        pass


clf = Mean_Shift()
clf.fit(X)
centroids = clf.centroids
print(centroids)
plt.scatter(X[: ,0], X[: ,1])
for c in centroids:
    plt.scatter(centroids[c][0],centroids[c],[1], color = 'k')


plt.show()

'''             
centroids = {}
       
for i in range(len(X)):
    centroids[i] = X[i]
    while True:
        new_centroids = []
        for i in centroids:
            in_bandwidth = []
            centroid = centroids[i]
            for featureset in X:
                if np.linalg.norm(featureset - centroid) < 4 :
                    in_bandwidth.append(featureset)

                new_centroid = np.average(in_bandwidth, axis = 0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))

'''        




























