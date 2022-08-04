import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import boston_housing
from tf_utils.dummyData import regression_data


if __name__ == "__main__":

    x, y = regression_data()
    x = x.reshape(-1, 1)
    print(x.shape)

    # x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    regr = LinearRegression()
    regr.fit(x_train, y_train)

    score = regr.score(x_test, y_test)
    print(f"R2-Score: {score}")
    print(f"Coeffs: {regr.coef_}")
    print(f"Intercep: {regr.intercept_}")

    y_pred = regr.predict(x_test)

    plt.scatter(x, y)
    plt.plot(x_test, y_pred)
    plt.show()
    print("Programm ENDE")
