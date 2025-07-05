import sklearn
import pandas as pd
import numbers as np
import matplotlib.pyplot as plt


def marvellouspredictior():
    #load the data /dataset
    x=[1,2,3,4,5]
    y=[3,4,2,4,5]
    print("Values of independent values is :",x)
    print("Values of dependent values is :",y)

    x_sum=0
    y_sum=0
    for i in range(len(x)):
        x_sum=x_sum+x[i]
        y_sum=y_sum+y[i]
    
    mean_x=x_sum/len(x)
    mean_y=y_sum/len(y)
    print("x_mean is",mean_x)
    print("y_mean is",mean_y)


def main():
    marvellouspredictior()


if __name__ == "__main__":
    main()