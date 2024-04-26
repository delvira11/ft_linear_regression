import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression





class FtLinearRegression:
    
    def __init__(self):
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.error_data = []
        X_values = 0
        y_values = 0
        real_x_values = 0
        real_y_values = 0


    def __loss_function(self, X, y, theta0, theta1):
        
        total_error = 0

        for i in range(len(X)):
            total_error += abs(y[i] - (theta0 + (theta1 * X[i])))
        return total_error

    def __gradient_descent(self, X, y, theta0, theta1, L):
        prediction_loss = 0 #Sum of all the real values - predicted values
        loss_per_Xi = 0 #Sum of all  real values - predicted values * Xi

        for i in range(len(X)):
            prediction_loss += ((theta0 + (theta1 * X[i])) - y[i])
            loss_per_Xi += (theta0 + (theta1 * X[i]) - y[i]) * X[i]
            
        theta0 -= (1 / len(X)) * L * prediction_loss
        theta1 -= (1 / len(X)) * L * loss_per_Xi
        return theta0[0], theta1[0]

    def __predict_n(self, i):
        
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        scaler_x.fit_transform(self.real_X_values.reshape(-1, 1))
        scaler_y.fit_transform(self.real_y_values.reshape(-1, 1))

     
        t_scaled = scaler_x.transform(np.array(i).reshape(-1, 1))
        print(t_scaled[0][0])
        predict = self.theta0 + (self.theta1 * t_scaled[0][0])
        print(predict)
        predicted_value = scaler_y.inverse_transform(np.array(predict).reshape(-1, 1))
        print(predicted_value)
        
        
        
        return predicted_value[0][0]

    def plot(self):
        
        scaler = StandardScaler()
        X = np.array(self.real_X_values).reshape(-1, 1)
        y = np.array(self.real_y_values).reshape(-1, 1)
        
        
        plt.scatter(X, y)
        plt.plot([X.min(), X.max()], [self.__predict_n(float(X.min())), self.__predict_n(float(X.max()))], color="red")
        
        lin_reg = LinearRegression()
        lin_reg.fit(X, y)
        print( "A : ", lin_reg.coef_)
        
        pricemin = lin_reg.predict([[X.min()]])
        pricemax = lin_reg.predict([[X.max()]])
        
        plt.plot([X.min(), X.max()], [pricemin.reshape(-1, 1)[0][0], pricemax.reshape(-1, 1)[0][0]], 'r--', color="green")
        plt.show()
        
    def fit(self, X, y):
        L = 0.05 # Learning rate
        self.error_data = []
        prev_theta0 = 1
        prev_theta1 = 1

        self.real_X_values = X
        self.real_y_values = y

        
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        X = scaler_x.fit_transform(X.reshape(-1, 1))
        y = scaler_y.fit_transform(y.reshape(-1, 1))
        
        #self.X_values = X
        #self.y_values = y
        while (round(self.theta0, 4) != round(prev_theta0, 4)) or (round(self.theta1, 4) != round(prev_theta1, 4)):
            prev_theta0, prev_theta1 = self.theta0, self.theta1
            self.theta0, self.theta1 = self.__gradient_descent(X, y, self.theta0, self.theta1, L)
            self.error_data.append(self.__loss_function(X ,y, self.theta0, self.theta1))
        
        # i = 150000
        # t_scaled = scaler_x.transform(np.array(i).reshape(-1, 1))
        # print(t_scaled[0][0])
        # predict = self.theta0 + (self.theta1 * t_scaled[0][0])
        # print(predict)
        # predicted_value = scaler_y.inverse_transform(np.array(predict).reshape(-1, 1))
        # print(predicted_value)
        #theta1_scaled = scaler_y.inverse_transform(np.array(y).reshape(-1, 1))
        #print(self.theta0, self.theta1)
        #print("scaled: ", theta0_scaled, theta1_scaled)
        
                    
    def plot_loss(self):
        plt.plot(self.error_data)
        plt.show()    
    def coef(self):
        return (self.theta0, self.theta1)
    def predict(self, number):
        scaler = StandardScaler()
        scaler.fit(self.real_X_values.reshape(-1, 1))
        scaled_number = scaler.transform(np.array(number).reshape(-1, 1))
        return self.theta0 + self.theta1 * scaled_number


import pandas as pd

lin_reg = FtLinearRegression()

data = pd.read_csv("data.csv")

X = np.array([data['km']])
y = np.array([data['price']])


lin_reg = FtLinearRegression()
lin_reg.fit(X, y)
lin_reg.plot()
#print(lin_reg.coef())
#print(lin_reg.predict(6000))


#lin_reg = LinearRegression()
#lin_reg.fit(np.array(data['km']).reshape(-1, 1), np.array(data['price']).reshape(-1, 1))
#print(lin_reg.coef_)