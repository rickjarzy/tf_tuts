
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.initializers import Constant
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from typing import Tuple
from tensorflow.keras.utils import to_categorical

def get_dataset(num_features: int, num_classes: int) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    bilder vom mnist dataset sind im (28,28,1) format - mein Dense layer will aber einen 1d array haben
    für die x daten: (28,28,1) --> (784, 1)
    für die y daten: (1,) --> (10,) will einen one-hot-vector mit den 10 klassen (bilder von buchstaben enthalten die hangeschriebenen zahlen von 0 bis 9)
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, num_features).astype(np.float32)
    x_test = x_test.reshape(-1, num_features).astype(np.float32)

    y_train = to_categorical(y_train, num_classes=num_classes, dtype=np.float32)
    y_test = to_categorical(y_test, num_classes=num_classes, dtype=np.float32)

    print(f"xtrain shape:  {x_train.shape}")
    print(f"ytrain shape:  {y_train.shape}")
    print(f"xtest shape:  {x_test.shape}")
    print(f"xtest shape:  {y_test.shape}")

    return (x_train, y_train), (x_test, y_test)


def build_model(num_features: int, num_classes: int) -> Sequential:
    """
    num_features ... die anzahl wie viele features hat unser x
    num_classes ... die output dimensionsgröße, shape des y das für das training verwendet wird

    init_w ... initialisiere meine Gewichtsmatrix mit RandomUniform in den Dense Layern - kernel_initializer
    Biasvektor möchte ich constant mit null initialisieren - > Constant - bias_initializer

    """

    init_w = RandomUniform(minval=-1.0, maxval=1.0)
    init_bias = Constant(value=0.0)

    model = Sequential()        # die layer werden sequentiel abgearbeitet

    # input layer
    # - units sind die anzahl der neuronen im hidden layer . war eine frei gewählte zahl
    # input_shape - die anzahl der features im unput layer
    # gewichtsmatrix vom input layer zum hidden layer is dann vom shape (input_feat.shape, 16)!!!
    model.add(Dense(units=500, kernel_initializer=init_w, bias_initializer=init_bias, input_shape=(num_features,)))

    # aktivierungsfuntion - die schicht vor der aktivvierungsfunktion hat immer die gleiche shape wie die aktivierungsfunktion selbst!!!
    model.add(Activation("relu"))                           # shape der gewichte vom hidden layer zum output sind (16,1)

    # Hidden layer --> hat keinen input shape mehr !!!
    model.add(Dense(units=250, kernel_initializer=init_w, bias_initializer=init_bias))
    model.add(Activation("relu"))                           # shape der gewichte vom hidden layer zum output sind (16,1)
    model.add(Dense(units=num_classes))  	    # output layer ( für regressionsaufgaben)

    # output layer is bei classifikations aufgaben immer eine softmax funktion --> die gibt an wie viele prozent eine klasse klassifiziert wurde
    model.add(Activation("softmax"))
    model.summary()
    return model


def main() -> None:

    # diese parameter sind durch das datenset vorgegeben -> (28,28,1)--> (783,1) grosser feature vector
    num_features = 784
    num_classes = 10


    (x_train, y_train), (x_test, y_test) = get_dataset(num_features, num_classes)

    model = build_model(num_features, num_classes)

    # gewichtsmatrix erstellen
    model.compile(
        loss="categorical_crossentropy",        # für classifikationsproblemen mit mehr als zwei klassen
        optimizer="Adam",           # Is eine Variante die gewisse probleme des SGD - Stochastic Gradiant Descent versucht auszumerzen - Adam bezieht Gradienten vorangegangener Batch Epochen TEILWEISE mit ein - lernrate wird auch mit der zeit angepasst
        metrics=["accuracy"]                             # die gängigste metric die es gibt
    )

    # training starten - ich habe 404 datenpunkte
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=128,         # mein datenset unterteilt sich somit in 4 batches zu (128,128,128,20) datenpunkten
        verbose=1,
        epochs=10,
        validation_data=(x_test, y_test))

    # auswertung des trainings
    scores = model.evaluate(
        x=x_test,
        y=y_test,
        verbose=0)

    print(scores)


if __name__ == "__main__":

    main()
    print("Programm ENDE")
