from sklearn .datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def marvellousdecisiontree():

    iris=load_iris() #load data

    x=iris.data
    y=iris.target

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

    model=DecisionTreeClassifier()

    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)

    accuracy=accuracy_score(y_test,y_pred)

    print("accuracy is :",accuracy*100)
    plt.figure(figsize=(12,10))
    plot_tree(model,filled=True,feature_names=iris.feature_names,class_names=iris.target_names)
    plt.title("Marvellous Decision Tree classifier")
    plt.show()


def main():
    marvellousdecisiontree()

if __name__ == "__main__":
    
    main()

