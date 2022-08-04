import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_boston



if __name__ == "__main__":

    dataset = load_boston()
    x = dataset.data
    y = dataset.target

    df = pd.DataFrame(x, columns=dataset.feature_names)
    # füge y hinzu - das is der preis pro quadrat fuss, den wir eigentlich predizieren wollen
    df["y"] = y
    print(df.head(n=10))
    print(df.info())
    print(df.describe())

    df.hist(bins=30, figsize=(15, 15))
    plt.show()
    print("Programm ENDE")
