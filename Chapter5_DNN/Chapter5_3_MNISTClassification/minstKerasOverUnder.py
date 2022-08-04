from operator import le
from typing import Tuple

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.initializers import Constant
from tensorflow.keras.initializers import RandomUniform, TruncatedNormal
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD


from tf_utils.plotting import display_convergence_error
from tf_utils.plotting import display_convergence_acc


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
    #init_w = HeNormal()

    # hyperparameter - welche verteilung - default parameter sind aber extrem gut, muss man nicht unbedingt ändern
    init_w = TruncatedNormal(mean=0.0, stddev=0.01)

    # hyperparameter
    init_b = Constant(value=0.0)

    model = Sequential()
    # hyperparameter - units, welche initialiserung ich pro schicht verwende
    # hyperparameter - wie viele Dense Layer will ich den verwenden??
    model.add(Dense(units=1600, kernel_initializer=init_w, bias_initializer=init_b, input_shape=(num_features,)))
    # hyperparameter - kann mich für jede schicht eigentlich entscheiden welche aktivierungsfunktion ich verwende
    model.add(Activation("relu"))
    model.add(Dense(units=400, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("relu"))
    model.add(Dense(units=200, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("relu"))
    model.add(Dense(units=100, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("relu"))
    model.add(Dense(units=num_classes, kernel_initializer=init_w, bias_initializer=init_b,))

    # hyperparameterm - softmax is standard als ausgabefunktion bei klassifikationen, muss man sich gut überlegen was den ( vor allem zum fall) besser passen würde
    model.add(Activation("softmax"))
    model.summary()
    return model


def main() -> None:

    # werden vom datenset vorgeschrieben
    num_features = 784
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = get_dataset(num_features, num_classes)

    model = build_model(num_features, num_classes)

    # Hyperparameter - optimizer selbst, als auch lernrate selber, obwohl die lernrate initial immer schon super is
    opt = Adam(learning_rate=0.0009)
    # opt = SGD(learning_rate=0.002)
    # opt = RMSprop(learning_rate=0.002)

    model.compile(
        loss="categorical_crossentropy",            # fehlerfunktion muss man nicht unbedingt ändern
        optimizer=opt,
        metrics=["accuracy"]
    )

    model.fit(
        x=x_train,
        y=y_train,
        epochs=20,                  # hyper parameter
        batch_size=256,             # hyper parameter
        verbose=1,
        validation_data=(x_test, y_test)
    )

    scores = model.evaluate(
        x=x_test,
        y=y_test,
        verbose=0
    )
    print(f"Scores on test set: {scores}")


if __name__ == "__main__":
    main()
