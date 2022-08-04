import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tf_utils.dummyData import regression_data


def mae(y_true, y_pred) -> np.ndarray:

    #mae_val = mean_absolute_error(y_true, y_pred)
    mae_val = np.mean(np.abs(y_true - y_pred))
    return mae_val


def mse(y_true, y_pred) -> np.ndarray:
    mse_val = np.mean(np.square(y_true - y_pred))
    #mse_val = mean_squared_error(y_true, y_pred)

    return mse_val

if __name__ == "__main__":
    x, y = regression_data()
    x = x.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    regr = LinearRegression()
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    r2_score = regr.score(x_test, y_test)
    mae_score = mae(y_test, y_pred)
    mse_score = mse(y_test, y_pred)

    print(f"R2-Score: {r2_score}")
    print(f"MAE: {mae_score}")
    print(f"MSE: {mse_score}")

    plt.scatter(x, y)
    plt.plot(x_test, y_pred)
    plt.show()
