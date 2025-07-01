import pandas as pd
from sklearn.model_selection import train_test_split

def loaddata(file_path):
    df=pd.read_csv(file_path)
    print("Datasets gets loaded in memory sucessfully")
    return df

def getinformation(df):
    print("Information abput loaded dataset is :")
    print("Shape of dataset :",df.shape)
    print("Columns ",df.columns)
    print("Missong Values : ",df.isnull().sum())


def encodedata(df):
    df["variety"]=df["variety"].map({"Setosa":0,"Versicolor":1,"Virginica":2})
    return df

def split_feature_target(df):
    x=df.drop('variety',axis=1)
    y=df['variety']
    return x,y

def split(x,y,size=0.2):
    return train_test_split(x,y,test_size=size)


def main():
    data=loaddata("iris.csv")
    print(data.head())

    getinformation(data)

    data=encodedata(data)
    print(data.head())
    
    print("Spliting features and lables")
    independenr,dependent=split_feature_target(data)
    print(independenr.head())
    print(dependent.head())


    x_train,x_test,y_train,y_test=split(independenr,dependent,0.3)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)


if __name__ == "__main__":
    main()