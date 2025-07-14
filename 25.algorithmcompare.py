from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def marvellouscalculateaccuracydecisiontree():
    iris=load_iris()
    data=iris.data
    target=iris.target


    x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2)

    model=tree.DecisionTreeClassifier()

    model.fit(x_train,y_train)
    prediction=model.predict(x_test)
    accuracy=accuracy_score(prediction,y_test)

    return accuracy

def main():
    result=marvellouscalculateaccuracydecisiontree()
    print("accuracy of decision tress is:",result*100)

if __name__ == "__main__":
    
    main()

