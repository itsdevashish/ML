from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def main():
    iris = load_iris()
    x = iris.data
    y = iris.target

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x[:,2], x[:,3], x[:,0], c=y, cmap="viridis", edgecolor='k')

    ax.set_xlabel("Petal Length")
    ax.set_ylabel("Petal Width")
    ax.set_zlabel("Sepal Length")
    plt.title("Marvellous visualization for IRIS")

    plt.show()

if __name__ == "__main__":
    main()
