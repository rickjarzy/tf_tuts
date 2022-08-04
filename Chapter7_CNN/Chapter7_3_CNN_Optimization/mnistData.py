import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


class MNIST:
    def __init__(self, with_norm: bool = True):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # preporcess x data
        self.x_train_: np.ndarray = None
        self.y_train_: np.ndarray = None
        self.x_val_: np.ndarray = None
        self.y_val_: np.ndarray = None
        self.x_train = x_train.astype(np.float32)
        self.x_train = np.expand_dims(x_train, axis=-1)
        if with_norm:
            self.x_train = self.x_train / 255.
        self.x_test = x_test.astype(np.float32)
        self.x_test = np.expand_dims(x_test, axis=-1)
        if with_norm:
            self.x_test = self.x_test / 255.

        # Dataset attributes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.img_shape = (self.width, self.height, self.depth)
        self.num_classes = 10  # len(np.unique(self.y_train))
        self.val_size = 0
        self.train_splitted_size = 0
        # Preproccess y data

        self.y_train = to_categorical(y_train, num_classes=self.num_classes, dtype=np.float32)
        self.y_test = to_categorical(y_test, num_classes=self.num_classes, dtype=np.float32)

    def get_train_set(self) -> Tuple[np.ndarray, np.ndarray]:

        return self.x_train, self.y_train

    def get_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_test, self.y_test

    def get_splittet_train_val_set(self, validation_size: float = 0.33) -> Tuple:
        self.x_train_, self.x_val_, self.y_train_, self.y_val_ = train_test_split(self.x_train, self.y_train, test_size=validation_size)
        self.val_size = self.x_val_.shape[0]
        self.train_splitted_size = self.x_train_.shape[0]
        return self.x_train_, self.x_val_, self.y_train_, self.y_val_


    def data_augmentation(self, augment_size: int = 5_000) -> None:
        image_generator = ImageDataGenerator(
            rotation_range=5,
            zoom_range=0.05,
            width_shift_range=0.08,
            height_shift_range=0.08
        )
        # Fit the data generator
        image_generator.fit(self.x_train, augment=True)

        # Get random train images for the data augmentation
        rand_idxs = np.random.randint(self.train_size, size=augment_size)
        x_augmented = self.x_train[rand_idxs].copy()
        y_augmented = self.y_train[rand_idxs].copy()
        # plt.imshow(x_augmented[0, :, :, 0], cmap="gray")
        # plt.show()
        x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size), batch_size=augment_size, shuffle=False).next()[0]
        # plt.imshow(x_augmented[0, :, :, 0], cmap="gray")
        # plt.show()

        # Append augmented images to the train set
        self.x_train = np.concatenate((self.x_train, x_augmented))
        self.y_train = np.concatenate((self.y_train, y_augmented))
        self.train_size = self.x_train.shape[0]


def main():
    data = MNIST()
    print(data.test_size)
    print(data.train_size)
    print(f"Max of x train: {np.max(data.x_train)}")
    print(f"Min of x train: {np.min(data.x_train)}")
    print(data.x_train.shape)
    print(data.y_train.shape)
    data.data_augmentation(augment_size=5_000)
    print(data.x_train.shape)
    print(data.y_train.shape)


if __name__ == "__main__":
    main()
