from sklearn.datasets import load_iris

def main():
    dataset=load_iris() #load zala dataset
    print("independent variable names are ")
    print(dataset.feature_names)


    print("Dedependent variable names are ")
    print(dataset.target_names)

if __name__ == "__main__":
    main()