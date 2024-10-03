from FtLinearRegression import FtLinearRegression
import numpy as np
import pandas as pd

if __name__ == '__main__':
    try:
        data = pd.read_csv("data.csv")
        X = np.array(data.iloc[:,0])
        y = np.array(data.iloc[:,1])
        lin_reg = FtLinearRegression()
        lin_reg.fit(X, y)
        
        error = 0
        for i in range(0, len(X)):
            prediction = lin_reg.predict(X[i])
            error += abs(prediction - y[i])
        print(error / len(X))
    except:
        print('Something went wrong')