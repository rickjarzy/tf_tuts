from sklearn.preprocessing import label_binarize
from tensorflow.keras.datasets import mnist
from tf_utils.plotting import display_digit


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print(f"x_train : {x_train.shape}")
    print("y_train : ", y_train.shape)
    print("x_test : ", x_test.shape)
    print("y_test : ", y_test.shape)

    for i in range(3):
        display_digit(x_train[i], label=y_train[i])
