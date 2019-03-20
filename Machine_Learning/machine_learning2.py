import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

fruits = pd.read_table("https://storage.googleapis.com/kaggle-datasets/9590/13660/fruit_data_with_colors.txt?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1545439947&Signature=CEOymPUDOnYJphr70jEtLTQZbnK5%2FN20e%2FMQg9C%2FtlpE2csbBGlMeIF249jQaP7mTPx%2BTSaaxfhrVEB9%2BpWE4hcmOjMMFZ3KR1Xn9tZVeB0UzXoKZVtNyzIVc6we4DqrQTHmZzx9OM1N97Jb2vVEpMxVfOzqQ0%2BNMHLgAjPqhwK2Gen9NchAt%2Fw8D2R1KQdHMcfWzBcqFjqzJxdd2lTO09noYmeaL9yPOOOeduJ%2BwDcjS%2F4Dg2MNH8CSv1Aox%2FAVvCd%2BgPhSx36U27HgPU2ON7GBpOEsFtnycyPenPOE4ifikW0hD4Rah1IKBjVv%2FikY17eBl%2FhbJES7gqOrGC2ugQ%3D%3D")

x = fruits[['mass','width','height']]
y = fruits['fruit_label']



"for taining set"
x_train, x_test, y_train, y_test =train_test_split(x, y, random_state=0)


x_plot = []
y_plot = []
num = 1
while num<43:
    knn = knn = KNeighborsClassifier(n_neighbors = num)

    y_plot.append(num)

    knn.fit(x_train, y_train)

    this = (knn.score(x_test, y_test))

    x_plot.append(this)

    num = num + 1
    

plt.plot(x_plot, y_plot)
plt.show()
"compute the accuracy"


'''
"predict use x.predict([[]])"

print(knn.predict([[20,4.3,5.5]]))


plt.scatter(x_train['width'], x_train['height'], c = y_train, marker = 'o', s=100)



plt.show()
'''
