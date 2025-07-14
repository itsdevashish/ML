from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def marvellouscalculateaccuracyKNN():
    iris=load_iris()
    data=iris.data
    target=iris.target

    x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=42)

    model=KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train,y_train)
    prediction=model.predict(x_test)
    accuracy=accuracy_score(prediction,y_test)
    print(accuracy*100)

    model=KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train,y_train)
    prediction=model.predict(x_test)
    accuracy=accuracy_score(prediction,y_test)
    print(accuracy*100)

    model=KNeighborsClassifier(n_neighbors=7)
    model.fit(x_train,y_train)
    prediction=model.predict(x_test)
    accuracy=accuracy_score(prediction,y_test)
    print(accuracy*100)

def main():
    result=marvellouscalculateaccuracyKNN()


if __name__ == "__main__":
    main()


