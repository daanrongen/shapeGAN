from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.interpolate import interp1d
import math

from helpers import clip, generatePolygon

# TO DO
# 1. improve polygon method (D) COMPLETED
# 2. write all training data (30.000 polygons) (D)
# 3. set up UAL GPU env (D) COMPLETED
# 4. set up DCGAN (H)
# 5. train (GAN)

# 128X128
WIDTH = 128
HEIGHT = 128
NUM_IMAGES = 80000


def save_image(verts, idx=0):
    im = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.polygon(
        verts,
        outline=(0, 0, 0),
        fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
    )
    im.save(f"polygons/{str(idx)}.png")


def create_n_shapes(n):
    for i in range(n):
        verts = generatePolygon(
            ctrX=(WIDTH / 2 + random.randint(-WIDTH / 4, WIDTH / 4)),
            ctrY=(HEIGHT / 2 + random.randint(-HEIGHT / 4, HEIGHT / 4)),
            aveRadius=WIDTH / 4,
            irregularity=random.uniform(0.1, 0.6),
            spikeyness=0.1,
            numVerts=20,
        )

        x = [x[0] for x in verts]
        y = [y[1] for y in verts]
        t = np.arange(len(x))
        ti = np.linspace(0, t.max(), 10 * t.size)

        xi = interp1d(t, x, kind="cubic")(ti)
        yi = interp1d(t, y, kind="cubic")(ti)

        verts = list(zip(xi, yi))
        save_image(verts, i)


create_n_shapes(NUM_IMAGES)
