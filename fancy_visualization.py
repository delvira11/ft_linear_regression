import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('ds.csv')
x = data['YearsExperience']
y = data['Salary']


def cost_func(x, y, m, b, n):
    cost = 0
    for i in range(n):
        cost += (y[i] - (m * x[i] + b)) ** 2
    cost_final = cost / n
    return cost


def grad_desc(x, y, m, b, n, L, d_m, d_b):

    for i in range(n):   
        d_m += -(2/n) * x[i] * (y[i] - (m * x[i] + b))
        d_b += -(2/n) * (y[i] - (m * x[i] + b))

    m = m - d_m * L
    b = b - d_b * L
    return m, b

d_m = 0
d_b = 0
L = 0.02
epochs = 700
m = 0
b = 0
n = len(x)
lin_reg = LinearRegression()
lin_reg.fit(data.YearsExperience.values.reshape(-1, 1), data.Salary.values)
x_min_max = np.array([[data.YearsExperience.min()], [data.YearsExperience.max()]])
y_train_pred = lin_reg.predict(x_min_max)
for i in range(epochs):
    plt.clf()
    plt.plot(x_min_max, y_train_pred, "g--")
    plt.plot(list(range(1, 12)), [m * x + b for x in range(1, 12)], color="red")
    m, b = grad_desc(x, y, m, b, n, L, d_m, d_b)
    plt.plot(np.array(data.YearsExperience), np.array(data.Salary), "b.")
    plt.draw()
    plt.pause(0.000001)
    print(i)
plt.show()