from typing import Tuple
import os
from matplotlib.ft2font import KERNING_UNSCALED
import matplotlib.pyplot as plt
import numpy as np
from regex import X
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import mnist
from tensorflow.keras.initializers import Constant
from tensorflow.keras.initializers import RandomUniform, TruncatedNormal
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from mnistData import MNIST

LOGS_DIR = os.path.abspath(r"C:/Users/parzberg/Entwicklung/Python/Tuts/udemy/tensorflow/logs")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)

MODEL_LOG_DIR = os.path.join(LOGS_DIR, "cnn_w_norm_wo_aug")


def build_model(img_shape: Tuple[int, int, int], num_classes: int) -> Model:

    input_img = Input(shape=img_shape)
    x = Conv2D(filters=16, kernel_size=3, padding="same", input_shape=img_shape)(input_img)
    x = Activation("relu")(x)
    x = Conv2D(filters=16, kernel_size=3, padding="same", input_shape=img_shape)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=32, kernel_size=3, padding="same", input_shape=img_shape)(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=3, padding="same", input_shape=img_shape)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=64, kernel_size=3, padding="same", input_shape=img_shape)(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=64, kernel_size=3, padding="same", input_shape=img_shape)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(units=128)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(x)

    model = Model(
        inputs=[input_img],
        outputs=[y_pred]
    )
    model.summary()

    return model


def main() -> None:
    data = MNIST()

    x_train_, x_val_, y_train_, y_val_ = data.get_splittet_train_val_set()


    model = build_model(data.img_shape, data.num_classes)

    model.compile(
        loss="categorical_crossentropy",
        optimizer="Adam",
        metrics=["accuracy"]
    )

    tb_callback = TensorBoard(
        log_dir=MODEL_LOG_DIR,
        write_graph=True
    )

    model.fit(
        x=x_train_,
        y=y_train_,
        epochs=10,
        batch_size=256,
        verbose=1,
        validation_data=(x_val_, y_val_),
        callbacks=[tb_callback])


def plot_filters(model: Model) -> None:
    first_conv_layer = model.layers[3]
    layer_weights = first_conv_layer.get_weights()
    kernels = layer_weights[0]

    num_filters = kernels.shape[3]
    subplot_grid = (num_filters // 4, 4)

    fig, ax = plt.subplots(subplot_grid[0], subplot_grid[1], figsize=(20, 20))
    ax = ax.reshape(num_filters)

    for filter_idx in range(num_filters):
        ax[filter_idx].imshow(kernels[:, :, 0, filter_idx], cmap="gray")

    ax = ax.reshape(subplot_grid)
    fig.subplots_adjust(hspace=0.5)
    plt.show()

    print("Layer weights: ", layer_weights)
    print("kernals shape: ", kernels.shape)

if __name__ == "__main__":
    main()
