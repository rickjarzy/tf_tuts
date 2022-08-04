import os
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import center_of_mass


FILE_PATH = os.path.abspath(__file__)
PROJECT_DIR = os.path.dirname(os.path.dirname(FILE_PATH))


def load(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def resize(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (28, 28))
    return image


def normalize(image: np.ndarray) -> np.ndarray:
    # gezeichnetes Bild is schwarze schrift und weisser hintergrund - im mnist datensatz is es aber umgekehrt deshalb
    # muss das bild invertiert werden - damit auch schwarzer hintergrund und weisse schrift vorliegt
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return image


def center(image: np.ndarray) -> np.ndarray:
    cy, cx = center_of_mass(image)
    rows, cols = image.shape

    shift_x = np.round(cols/2 - cx).astype(int)
    shift_y = np.round(rows/2 - cy).astype(int)

    Translations_matrix =np.array([[1,0, shift_x], [0,1,shift_y]]).astype(np.float32)
    image = cv2.warpAffine(image, M, (cols, rows))

def get_image(DrawingFrame: Any, debug: bool = False) -> np.ndarray:
    pixmap = DrawingFrame.grab()
    temp_image_file_path = os.path.join(PROJECT_DIR, "ressources", "imgs", "temp_img.jpg")
    pixmap.save(temp_image_file_path)

    image = load(temp_image_file_path)
    image = resize(image)
    image = normalize(image)
    image = center(image)
