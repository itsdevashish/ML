import pandas as pd
import matplotlib.pyplot as pt

def main():
    df=pd.read_csv("iris.csv")

    pt.hist(df["sepal.length"],bins=10,color="skyblue",edgecolor="black")
    pt.xlabel("Sepal Length")
    pt.ylabel("Frequency")
    pt.title("Marvellous Infosystems for IRIS")
    pt.show()


if __name__ == "__main__":
    main()