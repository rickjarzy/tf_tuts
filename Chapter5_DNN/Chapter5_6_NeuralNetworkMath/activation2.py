import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def f(x):
    return x**2 + x + 10

x = np.linspace(start=-10, stop=10, num=1000).reshape(-1, 1)
y = f(x)

def relu(x):
    if x > 0 : return x
    else: return 0


model = Sequential()
model.add(Dense(12))            # Input zu hidden
model.add(Activation("relu"))   # ReLu vom Hidden
model.add(Dense(1))             # Com Hidden zu Out
model.compile(optimizer=Adam(lr=1e-2), loss="mse")
model.fit(x,y, epochs=0)

#  gewiichte und bias vom Input Layer hinzu
W, b = model.layers[0].get_weights()
W2, b2 = model.layers[2].get_weights()    # an stelle 1 is die relu
print(W.shape, b.shape)
print(W2.shape, b2.shape)
