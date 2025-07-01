from sklearn.metrics import confusion_matrix

def main():
    #1=positive
    #0=negative
    
    acutal=[1,0,1,1,0,1,0,1,1,0]
    predicited=[1,0,1,0,0,1,1,1,1,1]

    con_mat=confusion_matrix(acutal,predicited)

    print("Confusuion matrix is :")
    print(con_mat)

    # TN.     FP
    # FN      TP

    # [2 2]
    # [1 5]

    #accouracy=(TN+FP)

if __name__ == "__main__":
    main()