import pandas as pd
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

df = pd.read_csv('breast_can_data.txt')


"filling in the gap, -9999 is for outliers"
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)


X = np.array(df.drop(['classes'],1))
y = np.array(df['classes'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)


accuracy = clf.score(X_test, y_test)


example_label = np.array([4,2,1,1,1,2,3,2,1])

'reshape is to make it into the right formate for sklearn'
example_label = example_label.reshape(1,-1) 
print(example_label)
prediction = clf.predict(example_label)
print(prediction)
print(accuracy)

'''
def dist(inp):
    this = 0 
    dis = 0
    lis = []
    
    for x in X:
        for j in range(len(x)):
            dis = (((inp[j])-(int(x[j])))**2) + dis
            #print(x[j])
            #print(inp[j])
        this = dis**.05
        lis.append(this)
        this =0 
        dis = 0
    thi = min(lis)
    po =(lis.index(thi))

    
    arr =range(1,699)
    so = (max(lis)+ thi)/3.5
    
    
    new_lis = []
    
    for a in range(len(arr)):
        if thi<=(lis[a])<=so:
            new_lis.append(lis[a])
    

    nnew_list = []
    for num in new_lis:
        nnew_list.append(y[(lis.index(num))])
    

    lab = y[po]
    acu = (nnew_list.count(y[po]))/len(nnew_list)*100
   
    
    
    return y[po], acu


print(dist([4,3,5,2,5,1,7,5,4]))
print(dist([1,1,5,2,2,1,7,1,3]))

'''
          
        

    
