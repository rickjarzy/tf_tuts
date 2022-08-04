from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os
from typing import Tuple
import numpy as np

MODELS_DIR = os.path.abspath(r"C:/Users/parzberg/Entwicklung/Python/Tuts/udemy/tensorflow/models")
MODEL_FILE_PATH = os.path.join(MODELS_DIR, "mnist_model.h5")
FULL_MODEL_FILES_PATH = os.path.join(MODELS_DIR, "full_mnist_model")

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




if __name__ == "__main__":
    num_features = 784
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = get_dataset(num_features, num_classes)

    model = load_model(filepath=FULL_MODEL_FILES_PATH)
    print(model.summary())

    scores = model.evaluate(
        x=x_test,
        y=y_test,
        verbose=0
    )
    print(f"scores after load: {scores}")

    print("Programm ENDE")
