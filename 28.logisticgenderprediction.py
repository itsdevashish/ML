import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def marvellouslogistic(datasetpath):

    df=pd.read_csv(datasetpath)

    print("Dimesion of dataframe is ",df.shape)
    print("Initial data is :",)
    print(df.head())

    df['Gender']=df['Gender'].map({'Female':0 ,'Male':1})
    print("Encoded data is :")
    print(df.head())

    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df,x='Height',y='Weight',hue='Gender',palette='Set1')
    plt.title("Marvellous Weight height")
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.show()

def main():
    marvellouslogistic("weight-height.csv")


if __name__ == "__main__":
    main()