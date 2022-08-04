import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def relu(x):
    if x > 0 : return x
    else: return 0


def sigmoid(x):
    return (1 / ( 1 + np.exp(-x)))

# y = ReLu (wx + b)
# y = Sigmoid(wx + b)
# shift der aktivierungsfunktion = -b/w
w = 1       # gewichtre
b = -4      # bias
act = sigmoid

x = np.linspace(start=-10, stop=10, num=1000)
y_act = np.array([act(xi * w + b) for xi in x])
y = np.array([act(xi * 1 + 0) for xi in x])         # standard


plt.figure(figsize=(8,5))
plt.grid(True)
plt.plot(x, y, color="blue", label="std")
plt.plot(x, y_act, color="red", label="own")
plt.legend()
plt.show()
