import matplotlib.pyplot as plt
from tf_utils.dummyData import regression_data, classification_data
import numpy as np


def model(x):
    m = -6.0     # slope
    b = 12.0     # intercept
    return m * x + b


if __name__ == "__main__":

    # x, y = regression_data()
    x, y = classification_data()
    y_pred = model(x)           # model vorhersage

    colors = np.array(["red", "blue"])
    plt.scatter(x[:, 0], x[:, 1], color=colors[y])
    plt.plot(x, y_pred)
    plt.show()
    print("Programm ENDE")
