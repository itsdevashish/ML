import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def marvellouspredictor():
    # Load the data / dataset
    x = [1, 2, 3, 4, 5]
    y = [3, 4, 2, 4, 5]

    print("Values of independent variables:", x)
    print("Values of dependent variables:", y)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    print("x_mean:", mean_x)
    print("y_mean:", mean_y)

    n = len(x)
    numerator = 0
    denominator = 0

    # y = mx + c
    for i in range(n):
        numerator += (x[i] - mean_x) * (y[i] - mean_y)
        denominator += (x[i] - mean_x) ** 2

    m = numerator / denominator
    print("Slope of line (m):", m)

    c = mean_y - (m * mean_x)
    print("Y-intercept (c):", c)

    # Regression line predictions
    x_np = np.array(x)
    y_pred = c + m * x_np

    # Plotting
    plt.scatter(x, y, color='red', label='Actual Data')
    plt.plot(x, y_pred, color='green', label='Regression Line')

    plt.xlabel("X: Independent Variable")
    plt.ylabel("Y: Dependent Variable")
    plt.title("Simple Linear Regression")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    marvellouspredictor()

if __name__ == "__main__":
    main()
