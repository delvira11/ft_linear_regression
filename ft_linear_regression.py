import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler




def loss_function(data, theta0, theta1):
    
    total_error = 0

    for i in range(len(data)):
        total_error += abs(data.iloc[i]["price"] - (theta0 + (theta1 * data.iloc[i]["km"])))

    print(total_error)
    return total_error


def gradient_descent(data, theta0, theta1, L):
    
    prediction_loss = 0 #Sum of all the real values - predicted values
    loss_per_Xi = 0 #Sum of all  real values - predicted values * Xi

    for i in range(len(data)):
        prediction_loss += ((theta0 + (theta1 * data.iloc[i]["km"])) - data.iloc[i]["price"])
        loss_per_Xi += (theta0 + (theta1 * data.iloc[i]["km"]) - data.iloc[i]["price"]) * data.iloc[i]["km"]
        
    theta0 -= (1 / len(data)) * L * prediction_loss
    theta1 -= (1 / len(data)) * L * loss_per_Xi

    return theta0, theta1




def main():
    theta0  = 0 # Constant term
    theta1 = 0 # Slope
    epochs = 1000
    L = 0.01 # Learning rate
    data = pd.read_csv("data.csv")
    scaler = StandardScaler()
    data["km"] = scaler.fit_transform(data[["km"]])
    for i in range(epochs):
        theta0, theta1 = gradient_descent(data, theta0, theta1, L)
        loss_function(data, theta0, theta1)

    
if __name__ == "__main__":
    main()