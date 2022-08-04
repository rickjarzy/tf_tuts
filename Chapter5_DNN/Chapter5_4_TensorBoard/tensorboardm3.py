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
from tensorflow.keras.models import load_model


MODELS_DIR = os.path.abspath(r"C:/Users/parzberg/Entwicklung/Python/Tuts/udemy/tensorflow/models")
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)
MODEL_FILE_PATH = os.path.join(MODELS_DIR, "mnist_model.h5")
FULL_MODEL_FILES_PATH = os.path.join(MODELS_DIR, "full_mnist_model")
LOGS_DIR = os.path.abspath(r"C:/Users/parzberg/Entwicklung/Python/Tuts/udemy/tensorflow/logs")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)

MODEL_LOG_DIR = os.path.join(LOGS_DIR, "mnist_model3")


def get_dataset(
    num_features: int,
    num_classes: int
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, num_features).astype(np.float32)
    x_test = x_test.reshape(-1, num_features).astype(np.float32)

    y_train = to_categorical(y_train, num_classes=num_classes, dtype=np.float32)
    y_test = to_categorical(y_test, num_classes=num_classes, dtype=np.float32)

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return (x_train, y_train), (x_test, y_test)


def build_model(num_features: int, num_classes: int) -> Sequential:
    #init_w = RandomUniform(minval=-1.0, maxval=1.0)
    init_w = TruncatedNormal(mean=0.0, stddev=0.01)
    #init_w = HeNormal()
    init_b = Constant(value=0.0)

    model = Sequential()
    model.add(Dense(units=1600, kernel_initializer=init_w, bias_initializer=init_b, input_shape=(num_features,)))
    model.add(Activation("elu"))
    model.add(Dense(units=400, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("elu"))
    model.add(Dense(units=200, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("elu"))
    model.add(Dense(units=100, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("elu"))
    model.add(Dense(units=num_classes, kernel_initializer=init_w, bias_initializer=init_b,))
    model.add(Activation("softmax"))
    model.summary()
    return model


def main() -> None:
    num_features = 784
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = get_dataset(num_features, num_classes)

    model = build_model(num_features, num_classes)

    model.compile(
        loss="categorical_crossentropy",
        optimizer="Adam",
        metrics=["accuracy"]
    )

    tb_callback = TensorBoard(log_dir=MODEL_LOG_DIR,
                            histogram_freq=1,
                            write_graph=True

    )


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
