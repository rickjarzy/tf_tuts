from pkgutil import get_data
from cv2 import detail_NoBundleAdjuster
import numpy as np
import sqlalchemy
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.initializers import Constant
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from typing import Tuple

from urllib3 import Retry


def get_dataset() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    x_train = x_train.astype(np.float32)
    y_train = y_train.reshape((-1, 1)).astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_test = y_test.reshape((-1, 1)).astype(np.float32)  # -1 bei reshape nimmt automatisch den ersten wert von im tuple des shape --> (404, 1)

    return (x_train, y_train), (x_test, y_test)


def build_model(num_features: int, num_target: int) -> Sequential:
    """
    num_features ... die anzahl wie viele features hat unser x
    num_target ... die output dimensionsgröße, shape des y das für das training verwendet wird

    init_w ... initialisiere meine Gewichtsmatrix mit RandomUniform in den Dense Layern
    Biasvektor möchte ich constant mit null initialisieren - > Constant

    """

    init_w = RandomUniform(minval=-1.0, maxval=1.0)
    init_bias = Constant(value=0.0)

    model = Sequential()        # die layer werden sequentiel abgearbeitet

    # input layer
    model.add(Dense(units=16, kernel_initializer=init_w, bias_initializer=init_bias, input_shape=(num_features,)))  # - units sind die anzahl der neuronen im hidden layer . war eine frei gewählte zahl
                                                             # input_shape - die anzahl der features im unput layer
                                                             # gewichtsmatrix vom input layer zum hidden layer is dann vom shape (input_feat.shape, 16)!!!

    # aktivierungsfuntion - die schicht vor der aktivvierungsfunktion hat immer die gleiche shape wie die aktivierungsfunktion selbst!!!
    model.add(Activation("relu"))                           # shape der gewichte vom hidden layer zum output sind (16,1)

    # output layer
    model.add(Dense(units=num_target))  	    # output layer
    model.summary()
    return model


def main() -> None:
    (x_train, y_train), (x_test, y_test) = get_dataset()
    print(f"xtrain shape:  {x_train.shape}")
    print(f"ytrain shape:  {y_train.shape}")
    print(f"xtest shape:  {x_test.shape}")
    print(f"xtest shape:  {y_test.shape}")

    num_features = 13       # nachschauen wie viele spalten/features hat mein trainingsset
                            # das sind meine 13 input neuronen
    num_targets = 1         # will als output nur eine zahl und zwar preis pro squarefeet

    model = build_model(num_features, num_targets)


if __name__ == "__main__":

    main()
    print("Programm ENDE")
