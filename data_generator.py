import numpy as np 
import pandas as pd
from sklearn.datasets import make_circles

def generate_synthetic_clusters():
    # read synthetic csv
    synthetic_df = pd.read_csv("data/synthetic.csv")

    # retrieve X
    X = np.column_stack((synthetic_df["4"], synthetic_df["5"]))

    # retrieve y
    y = np.array([1 if x == "yes" else 0 for x in synthetic_df["outlier"]])

    return X, y

def generate_synthetic_d1():
    X = []
    y = []

    # create bottom right rectangle
    for i in range(0, 21):
        for j in range(0, 9):
            X.append([i*0.5, j*0.5])
    
    # create top left rectangle
    for i in range(0, 41):
        for j in range(0, 17):
            X.append([i*0.1, 6+j*0.125])

    # create top right rectangle
    for i in range(0, 17):
        for j in range(0, 41):
            X.append([8+i*0.25/2, 5 + j*0.125])
            
    # random distribution
    np.random.seed(seed=123)
    X.extend(list(np.random.uniform((0, 4), (10, 10), (69, 2))))

    for point in X:
        point_x, point_y = point[0], point[1]
        if point_y <= 4: # in bottom rectangle
            y.append(0)
        elif point_x <= 4 and point_y >= 6 and point_y <= 8: # in top left rectangle
            y.append(0) 
        elif point_x >= 8 and point_y >= 5: # in top right rectangle
            y.append(0)
        else: # outlier
            y.append(1)

    return np.array(X), np.array(y)

def generate_synthetic_d2():
    def helper_y(x, m, b):
        '''
        equation of line
        '''
        return m*x + b

    # set seed
    np.random.seed(seed=0)

    # store X and y
    X = []
    y = []

    # generate line data
    x_line = np.linspace(10, 20, 400)
    y_line_above = [helper_y(x, 1, 0) + abs(np.random.normal(0,0.25)) for x in x_line]
    y_line_below = [helper_y(x, -1, 20) - abs(np.random.normal(0,0.25)) for x in x_line]

    # add line data
    for i in range(len(x_line)):
        X.append([x_line[i], y_line_above[i]])
        X.append([x_line[i], y_line_below[i]])
        y.append(0)
        y.append(0)
    
    # generate circular data
    X_circ, y_circ =  make_circles(n_samples=400, shuffle=True, noise=0.05, random_state=123, factor=0.5)
    X_circ = X_circ[y_circ == 1]
    X_circ = [[1+x[0], 10+x[1]] for x in X_circ]
    y_circ = y_circ[y_circ == 1]

    # add circular data
    X.extend(list(X_circ))
    y.extend([0]*len(X_circ))

    # generate 100 random points for noise
    X_rand = list(np.random.uniform((0, 0), (20, 20), (79, 2)))
    X.extend(list(X_rand))
    y.extend([1]*len(X_rand))

    return np.array(X), np.array(y)