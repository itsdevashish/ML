from sklearn import tree
#rough=0
#smooth=1
def main():
    #feartes 
    ballfeatures=[[35,0],[47,0],[90,1],[48,0],[90,1],[35,0],[92,1],[35,0],[35,0],[35,0]]
     
    #lables
    ballnames= ["tennis","tennis","cricket","tennis","cricket","tennis","cricket","tennis","tennis","tennis"]

    obj=tree.DecisionTreeClassifier()

    obj=obj.fit(ballfeatures,ballnames) #training
    print(obj.predict([93,1])) #testing

if __name__ == "__main__":
    main()