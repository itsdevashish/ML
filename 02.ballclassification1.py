from sklearn import tree


def main():
    #feartes 
    ballfeatures=[ [35,"rough"],[47,"rough"],[90,"smooth"],[48,"rough"],[90,"smooth"],[35,"rough"],[92,"smooth"],[35,"rough"],[35,"rough"],[35,"rough"]]
     
    #lables
    ballnames= ["tennis","tennis","cricket","tennis","cricket","tennis","cricket","tennis","tennis","tennis"]

    obj=tree.DecisionTreeClassifier()

    obj=obj.fit(ballfeatures,ballnames) #training
    print(obj.predict([93,"smooth"])) #testing

if __name__ == "__main__":
    main()