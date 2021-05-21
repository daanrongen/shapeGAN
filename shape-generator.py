import matplotlib.pyplot as plt
import numpy as np

# 256x256
WIDTH = 256
HEIGHT = 256

figure, axes = plt.subplots()
circle = plt.Circle((125, 125), 20.0)

axes.set_aspect(1)
plt.xlim(0, WIDTH)
plt.ylim(0, HEIGHT)
axes.add_artist(circle)
plt.show()


def random_shape():
    grid = np.array(256, 256)
