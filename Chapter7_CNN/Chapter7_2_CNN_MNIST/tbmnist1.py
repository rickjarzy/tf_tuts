from typing import Tuple
import os
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import mnist
from tensorflow.keras.initializers import Constant
from tensorflow.keras.initializers import RandomUniform, TruncatedNormal
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D


LOGS_DIR = os.path.abspath(r"C:/Users/parzberg/Entwicklung/Python/Tuts/udemy/tensorflow/logs")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)

MODEL_LOG_DIR = os.path.join(LOGS_DIR, "mnist_cnn1")


def get_dataset(
    num_classes: int
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype(np.float32)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = x_test.astype(np.float32)
    x_test = np.expand_dims(x_test, axis=-1)

    y_train = to_categorical(y_train, num_classes=num_classes, dtype=np.float32)
    y_test = to_categorical(y_test, num_classes=num_classes, dtype=np.float32)

    return (x_train, y_train), (x_test, y_test)


def build_model(img_shape: Tuple[int, int, int], num_classes: int) -> Sequential:

    model = Sequential()
    # filters=16 - bedeutet das 16 Filter angewandt werden
    model.add(Conv2D(filters=64, kernel_size=3, input_shape=img_shape))     # nunr input layer hat input_shape
    model.add(Activation("relu"))                       #  muss hier checken welcher shape rauskommt
    model.add(MaxPooling2D())                           # default values: pool_size= (2,2), strides=2, padding="valid"                                                          # man muss hier auf alle fälle behirnen dass die poolsize und stride zusammen passen sollten
                                                        # man muss hier auch behirnen das nach dem Maxpooling sich die grösse der Bildmatrix wieder ändert
    model.add(Conv2D(filters=32, kernel_size=3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D())

    model.add(Conv2D(filters=16, kernel_size=3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())                # macht aus Bildmatrix einen vektor -> rows*cols*3d_info_anzahl
    model.add(Dense(units=num_classes))
    model.add(Activation("softmax"))

    model.summary()
    return model


def main() -> None:
    img_shape = (28, 28, 1)
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = get_dataset(num_classes)

    model = build_model(img_shape, num_classes)

    model.compile(
        loss="categorical_crossentropy",
        optimizer="Adam",
        metrics=["accuracy"]
    )

    tb_callback = TensorBoard(log_dir=MODEL_LOG_DIR,
                            histogram_freq=1,
                            write_graph=True)

    model.fit(
        x=x_train,
        y=y_train,
        epochs=20,
        batch_size=256,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[tb_callback]
    )
    scores = model.evaluate(
        x=x_test,
        y=y_test,
        verbose=0
    )
    print(f"Scores before saving: {scores}")



if __name__ == "__main__":
    main()
