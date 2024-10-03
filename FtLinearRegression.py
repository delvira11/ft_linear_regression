import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression

class Ft_StandardScaler():
    def __init__(self):
        pass
    
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X - self.mean, axis=0)
        return self
    def transform(self, X):
        return (X - self.mean) / self.scale
    def fit_transform(self, X):
        fitted = self.fit(X)
        transformed = fitted.transform(X)
        return transformed
    def inverse_transform(self, X):
        return X * self.scale + self.mean


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


    def plot(self):
        
        scaler = Ft_StandardScaler()
        X = np.array(self.real_X_values).reshape(-1, 1)
        y = np.array(self.real_y_values).reshape(-1, 1)
        
        
        plt.scatter(X, y)
        plt.plot([X.min(), X.max()], [self.predict(float(X.min())), self.predict(float(X.max()))], color="red")
        
        # Scikit Learn plot
        # lin_reg = LinearRegression()
        # lin_reg.fit(X, y)
        # pricemin = lin_reg.predict([[X.min()]])
        # pricemax = lin_reg.predict([[X.max()]])
        # plt.plot([X.min(), X.max()], [pricemin.reshape(-1, 1)[0][0], pricemax.reshape(-1, 1)[0][0]], 'g--')

        plt.show()
        
    def fit(self, X, y):
        L = 0.05 # Learning rate
        self.error_data = []
        prev_theta0 = 1
        prev_theta1 = 1

        self.real_X_values = X
        self.real_y_values = y

        
        scaler_x = Ft_StandardScaler()
        scaler_y = Ft_StandardScaler()
        X = scaler_x.fit_transform(X.reshape(-1, 1))
        y = scaler_y.fit_transform(y.reshape(-1, 1))
        
        while (round(self.theta0, 4) != round(prev_theta0, 4)) or (round(self.theta1, 4) != round(prev_theta1, 4)):
            prev_theta0, prev_theta1 = self.theta0, self.theta1
            self.theta0, self.theta1 = self.__gradient_descent(X, y, self.theta0, self.theta1, L)
            self.error_data.append(self.__loss_function(X ,y, self.theta0, self.theta1))
                    
    def plot_loss(self):
        plt.plot(self.error_data)
        plt.show()

    def save_coef(self):
        x1, x2 = 0, 100
        y1 = self.predict(x1)
        y2 = self.predict(x2)
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
        else:
            raise Exception('can\'t divide by 0')
        const = y1
        return const, slope
    
    def predict(self, i):
        
        scaler_x = Ft_StandardScaler()
        scaler_y = Ft_StandardScaler()
        scaler_x.fit_transform(self.real_X_values.reshape(-1, 1))
        scaler_y.fit_transform(self.real_y_values.reshape(-1, 1))

     
        t_scaled = scaler_x.transform(np.array(i).reshape(-1, 1))
        predict = self.theta0 + (self.theta1 * t_scaled[0][0])
        predicted_value = scaler_y.inverse_transform(np.array(predict).reshape(-1, 1))
        return predicted_value[0][0]


    # def predict_scaled(self, number):
    #     scaler = Ft_StandardScaler()
    #     scaler.fit(self.real_X_values.reshape(-1, 1))
    #     scaled_number = scaler.transform(np.array(number).reshape(-1, 1))
    #     return self.theta0 + self.theta1 * scaled_number


if __name__ == '__main__':
    try:
        lin_reg = FtLinearRegression()

        data = pd.read_csv("data.csv")

        X = np.array(data.iloc[:,0])
        y = np.array(data.iloc[:,1])
        lin_reg = FtLinearRegression()
        lin_reg.fit(X, y)
        lin_reg.plot()
        lin_reg.plot_loss()
        results = lin_reg.save_coef()
        
        file = open('results.txt', 'w')
        file.write(str(results[0]))
        file.write(',')
        file.write(str(results[1]))
    except:
        print('Something went wrong')
