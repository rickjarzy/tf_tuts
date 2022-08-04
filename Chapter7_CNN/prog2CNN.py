import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import MaxPool2D


def max_pooling(image: np.ndarray) -> np.ndarray:
    rows, cols = image.shape
    print(rows, cols)
    print("rows, type: ", type(rows))
    max_pool_img = np.zeros((rows // 2, cols // 2))

    cou_row = 0
    for row in range(0, rows - 2, 2):
        cou_col = 0
        for col in range(0, cols - 2, 2):
            print(f"row: {row}, col: {col}")
            max_pool_kernel = image[row:row + 2, col:col + 2]
            max_pool_img[cou_row, cou_col] = np.max(max_pool_kernel)
            cou_col += 1
        cou_row += 1
    print(max_pool_img)
    print("POOOL Shape ", max_pool_img.shape)
    return max_pool_img  # TODO


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image = x_train[0]
    image = image.reshape((28, 28)).astype(np.float32)

    pooling_image = max_pooling(image)

    print(f"Prvious shape: {image.shape} current shape: {pooling_image.shape}")
    print(f"Pooled Image:\n{pooling_image.squeeze()}")

    layer = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')
    pooling_image_tf = layer(image.reshape((1, 28, 28, 1))).numpy()
    print(f"Pooled Image TF:\n{pooling_image_tf.squeeze()}")
    assert np.allclose(pooling_image.flatten(), pooling_image_tf.flatten())

    fig, axs = plt.subplots(nrows=1, ncols=3)
    axs[0].imshow(image, cmap="gray")
    axs[1].imshow(pooling_image, cmap="gray")
    axs[2].imshow(pooling_image_tf.squeeze(), cmap="gray")
    plt.show()
