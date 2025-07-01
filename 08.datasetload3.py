from sklearn.datasets import load_iris

def main():
    dataset=load_iris() #load zala dataset

    line="_"*30

    print("Elements form the datasets are")

    for i in range(len(dataset.target)):
        print("ID %d , Features %s, Label:%s " %(i,dataset.data[i],dataset.target[i]))
    print(line)

if __name__ == "__main__":
    main()