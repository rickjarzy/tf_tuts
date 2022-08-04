import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D


def conv2D(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    print("INSIDE CONV2D\n", image)
    print("shape: ", image.shape)
    output_mat = np.zeros(image.shape)
    padded_image = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    padded_image[1:-1, 1:-1] = image
    print("padded image: \n", padded_image)

    for rows in range(0, image.shape[0], 1):
        for cols in range(0, image.shape[1], 1):
            print("rows: ", rows, "cols: ", cols)
            print("image kernel: \n", padded_image[rows:rows + 3, cols:cols + 3])
            print("numpy mat: \n", np.multiply(padded_image[rows:rows + 3, cols:cols + 3], kernel))
            print("summed up: ", np.sum(np.multiply(padded_image[rows:rows + 3, cols:cols + 3], kernel)))
            output_mat[rows, cols] = np.sum(np.multiply(padded_image[rows:rows + 3, cols:cols + 3], kernel))
    print(output_mat)
    return output_mat  # TODO


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image = np.arange(16)
    image = image.reshape((4, 4)).astype(np.float32)
    kernel = np.ones(shape=(3, 3))

    conv_image = conv2D(image, kernel)

    print(f"Prvious shape: {image.shape} current shape: {conv_image.shape}")
    print(f"Conved Image:\n{conv_image.squeeze()}")

    layer = Conv2D(filters=1, kernel_size=(3, 3), strides=1, padding='same')
    layer.build((4, 4, 1))
    W, b = layer.get_weights()
    layer.set_weights([np.ones_like(W), np.zeros_like(b)])
    conv_image_tf = layer(image.reshape((1, 4, 4, 1))).numpy()
    print(f"Conved Image TF:\n{conv_image_tf.squeeze()}")
    not_equals = (conv_image.flatten() != conv_image_tf.flatten())
    # assert np.allclose(conv_image.flatten(), conv_image_tf.flatten())

    fig, axs = plt.subplots(nrows=1, ncols=3)
    axs[0].imshow(image, cmap="gray")
    axs[1].imshow(conv_image, cmap="gray")
    axs[2].imshow(conv_image_tf.squeeze(), cmap="gray")
    plt.show()
