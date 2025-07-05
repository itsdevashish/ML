import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def marvellouspredictior():
    #load the data /dataset
    x=[1,2,3,4,5]
    y=[3,4,2,4,5]
    print("Values of independent values is :",x)
    print("Values of dependent values is :",y)

    mean_x=np.mean(x)
    mean_y=np.mean(y)

    print("x_mean is",mean_x)
    print("y_mean is",mean_y)


    n=len(x)
    numerator=0
    denamenator=0

    #y=mx+c
    for i in range(n):
        numerator=numerator+(x[i]-mean_x)*(y[i]-mean_y)
        denamenator=denamenator + (x[i]-mean_x)**2

    m=numerator/denamenator
    print("Slope of line m is :",m)

    c=mean_y-(m*mean_x)
    print("Y intercept of line is :",c)


def main():
    marvellouspredictior()


if __name__ == "__main__":
    main()