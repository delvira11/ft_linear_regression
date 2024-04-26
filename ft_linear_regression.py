import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np




def loss_function(X, y, theta0, theta1):
    
    total_error = 0

    for i in range(len(X)):
        total_error += abs(y[i] - (theta0 + (theta1 * X[i])))
    return total_error


def gradient_descent(X, y, theta0, theta1, L):
    
    prediction_loss = 0 #Sum of all the real values - predicted values
    loss_per_Xi = 0 #Sum of all  real values - predicted values * Xi

    for i in range(len(X)):
        prediction_loss += ((theta0 + (theta1 * X[i])) - y[i])
        loss_per_Xi += (theta0 + (theta1 * X[i]) - y[i]) * X[i]
        
    theta0 -= (1 / len(X)) * L * prediction_loss
    theta1 -= (1 / len(X)) * L * loss_per_Xi

    return theta0[0], theta1[0]



def predict(theta0, theta1, km):
    price = theta0 + (theta1 * km)
    return price

def plot_line(X, y, theta0, theta1):
    plt.scatter(X, y)

    
    plt.plot([X.min(), X.max()], [predict(theta0, theta1, X.min()), predict(theta0, theta1, X.max())], color="red")
    
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    
    pricemin = lin_reg.predict([[X.min()]])
    pricemax = lin_reg.predict([[X.max()]])
    
    plt.plot([X.min(), X.max()], [pricemin.reshape(-1, 1)[0][0], pricemax.reshape(-1, 1)[0][0]], 'r--', color="green")
    plt.show()

def main():
    theta0  = 0 # Constant term
    theta1 = 0 # Slope
    #epochs = 1000
    L = 0.05 # Learning rate
    data = pd.read_csv("data.csv")
    
    X = data['km']
    y = data['price']
    
    error_data = []
    prev_theta0 = 1
    prev_theta1 = 1
    scaler = StandardScaler()
    X = scaler.fit_transform(data[["km"]])
    y = np.array(y)
    while (round(theta0, 2) != round(prev_theta0, 3)) and (round(theta1, 2) != round(prev_theta1, 3)):
        prev_theta0, prev_theta1 = theta0, theta1
        theta0, theta1 = gradient_descent(X, y, theta0, theta1, L)
        error_data.append(loss_function(X ,y, theta0, theta1))
    plot_line(X, y, theta0, theta1)
    plt.plot(error_data)
    plt.show()

if __name__ == "__main__":
    main()
    
    
