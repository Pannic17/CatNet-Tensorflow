import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

print(int(-255))

    count = 0
    fig.axis([0, 3, 0, 1])
    for center in centers:
        r, g, b = colorsys.hsv_to_rgb(center[0] / 180, center[1] / 255, center[2] / 255)
        color = [r, g, b]
        print([r * 255, g * 255, b * 255])
        # color = [center[0]/255, center[1]/255, center[2]/255]
        fig.add_patch(patches.Rectangle((count, 0), 1, 1, color=color))
        count += 1