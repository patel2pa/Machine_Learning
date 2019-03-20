from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random


style.use('fivethirtyeight')

#x = np.array([1,2,3,4,5,6], dtype = np.float64)
#y = np.array([5,4,6,5,6,7], dtype = np.float64)

def create_dataset(hm, variance, step = 2,correlation=False):
    val =1
    y = []
    for i in range (hm):
        ys = val+random.randrange(-variance, variance)
        y.append(ys)
        if correlation and correlation=='pos':
            val+=step
        elif correlation and correlation == 'neg':
            val -= step
    x = [i for i in range(len(y))]       
    return np.array(x,dtype = np.float64), np.array(y, dtype=np.float64)



x,y = create_dataset(40,10,2,correlation = 'pos')
    
    


def best_fit_slope_and_intercept(x,y):
    m = (((mean(x)*mean(y)) - mean(x*y))/((mean(x)*mean(x))-mean(x*x)))
    b = mean(y) - (m*mean(x))
    return m, b

def squared_error(y_orgin, y_line):
    return sum((y_line - y_orgin)**2)

def coefficient_of_determination(y_orgin, y_line):
    y_mean_line = [mean(y_orgin) for ys in y_orgin]
    squared_error_regr = squared_error(y_orgin,y_line)
    squared_error_y_mean = squared_error(y_orgin, y_mean_line)
    return 1-(squared_error_regr/squared_error_y_mean )
   






m,b = best_fit_slope_and_intercept(x,y)

regression_line = []


for xs in x:
    regression_line.append((m*xs)+b)

predict_x = 8
predict_y = (m*predict_x)+b

r_squared = coefficient_of_determination(y,regression_line)      
print(r_squared)
plt.scatter(x,y)
plt.scatter(predict_x,predict_y)
plt.plot(x,regression_line)
plt.show()


  



