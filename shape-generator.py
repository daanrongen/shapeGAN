import matplotlib.pyplot as plt
import random
import turtle
import math

from PIL import Image, ImageDraw
from helpers import clip, generatePolygon

# 256x256
WIDTH = 256
HEIGHT = 256
COLORS = [
    (255, 255, 255),
    (0, 255, 255),
    (0, 0, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 255, 0),
]


# https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon


def save_image(verts, idx=0):
    im = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.polygon(verts, fill=COLORS[random.randint(0, len(COLORS) - 1)])
    im.save(f"polygons/{str(idx)}.png")


def create_n_shapes(n):
    for i in range(n):
        verts = generatePolygon(
            ctrX=WIDTH / 2,
            ctrY=HEIGHT / 2,
            aveRadius=80,
            irregularity=random.uniform(0, 1),
            spikeyness=random.uniform(0, 0.5),
            numVerts=random.randint(80, 120),
        )

        save_image(verts, i)


create_n_shapes(5)
