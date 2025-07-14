import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

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

    X=df[['Height','Weight']]
    Y=df[['Gender']]

    x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.2,random_state=42) #spliting
    model=LogisticRegression()
    model.fit(x_train,y_train)
    Y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_test,Y_pred)
    print("Accuracy is :",accuracy*100)

    conf_matrix=confusion_matrix(y_test,Y_pred)
    print(conf_matrix)

    report=classification_report(y_test,Y_pred)
    print("Classification report is :")
    print(report)

def main():
    marvellouslogistic("weight-height.csv")


if __name__ == "__main__":
    main()