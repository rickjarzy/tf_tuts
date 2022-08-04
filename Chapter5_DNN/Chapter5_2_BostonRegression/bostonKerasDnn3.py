
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.initializers import Constant
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from typing import Tuple



def r_squared(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:

    """
    quelle und grafik R2 ... https://medium.com/analytics-vidhya/r-squared-formula-explanation-6dc0096ce3ba
    """
    error = tf.math.subtract(y_true, y_pred)        # rechne mir den fehler aus den eingegebenen daten und den predizierten daten
    squared_error = tf.math.square(error)
    numerator = tf.math.reduce_sum(squared_error)

    y_true_mean = tf.math.reduce_mean(y_true)
    mean_deviation = tf.math.subtract(y_true, y_true_mean)
    squared_mean_deviation = tf.math.square(mean_deviation)

    denominator = tf.reduce_sum(squared_mean_deviation)

    r2 = tf.math.subtract(1.0, tf.math.divide(numerator, denominator))

    # clippe negative werte falls das vorkommen sollte und setzte ihn dann auf 0
    r2_clipped = tf.clip_by_value(r2, clip_value_min=0.0, clip_value_max=1.0)

    return r2_clipped


def r_squared2(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:

    """
    quelle und grafik R2 ... https://medium.com/analytics-vidhya/r-squared-formula-explanation-6dc0096ce3ba
    """
    error = tf.math.subtract(y_true, y_pred)
    error_squared = tf.math.square(error)

    numerator = tf.math.reduce_sum(error_squared)   # zähler

    y_true_mean = tf.math.reduce_mean(y_true)   # nennen
    mean_diviation = tf.math.subtract(y_true, y_true_mean)
    squared_mean_deviation = tf.math.square(mean_diviation)
    denominator = tf.math.reduce_sum(squared_mean_deviation)

    r2 = tf.math.subtract(1.0, tf.math.divide(numerator, denominator))

    # abfangen der negativen werte - r2 kann die nämlich annehmen
    r2_cliped = tf.clip_by_value(r2, clip_value_min=0.0, clip_value_max=1.0)
    return r2_cliped


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

    init_w ... initialisiere meine Gewichtsmatrix mit RandomUniform in den Dense Layern - kernel_initializer
    Biasvektor möchte ich constant mit null initialisieren - > Constant - bias_initializer

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

    # gewichtsmatrix erstellen
    model.compile(
        loss="mse",
        optimizer="Adam",           # Is eine Variante die gewisse probleme des SGD - Stochastic Gradiant Descent versucht auszumerzen - Adam bezieht Gradienten vorangegangener Batch Epochen TEILWEISE mit ein
        metrics=[r_squared2]                         # ad optimizer - lernrate wird auch mit der zeit angepasst
    )

    # training starten - ich habe 404 datenpunkte
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=128,         # mein datenset unterteilt sich somit in 4 batches zu (128,128,128,20) datenpunkten
        verbose=1,
        epochs=3_000,
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
