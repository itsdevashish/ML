import pandas as pd
from sklearn.model_selection import train_test_split
def main():
    
    df=pd.read_csv("iris.csv")
    print("Data set loaded sucessfully")
 
    df['variety']=df["variety"].map({'Setosa' : 0 , 'Versicolor':1 , 'Virginica' :2})
    print(df['variety'])

    x=df.drop('variety',axis='columns')
    y=df['variety']


    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    print("Dimension of x_train",x_train.shape)
    print("Dimension of x_test",x_test.shape)
    print("Dimension of y_train",y_train.shape)
    print("Dimension of y_test",y_test.shape)
    

if __name__ == "__main__":
    main()