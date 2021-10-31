import colorsys
import os

import cv2
import math
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors


def detect(filename):
    # detect the cat face by opencv haarcascade
    # return the coordinates of cat face
    face_cascade = cv2.CascadeClassifier(
        'H://Project/21ACB/cat_face_cut/haarcascade_frontalcatface_extended.xml')
    # read image
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect face
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.02,
                                          minNeighbors=5)
    size = img.shape[0] * img.shape[1]

    # 绘制人脸矩形框
    for (x, y, w, h) in faces:
        w_delta = math.ceil(w * 0.1)
        h_delta = math.ceil(h * 0.1)
        # print(w, h)
        # print(x, y)

        face_size = w * h

        # change the coordinate, increase the rectangle by 10% for each side and move up to include the ears
        x1 = x - w_delta
        y1 = y - h_delta * 2
        x2 = x + w + w_delta
        y2 = y + h

        # draw the cat face, originate and cut
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # print(x1, y1, x2, y2)
        # return x1, y1, x2, y2

        coordinates = [x1, y1, x2, y2, w]
        positive = True
        for coordinate in coordinates:
            if coordinate < 0:
                positive = False
                break

        # test whether the detected area is valid
        if w < 30 or (face_size < (size * 0.01)):
            return None

        # return extended box if available
        if positive is True:
            return coordinates
        else:
            coordinates = [x, y, x + w, y + h, w]
            return coordinates


def cut(filename):
    # cut the cat face out according to the coordinates given by detection
    # return the image for saving
    coordinates = detect(filename)
    img = cv2.imread(filename)
    if coordinates is not None:
        new = img[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]
        return new, coordinates
    else:
        return None, None


def recolor(img):
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    return img


def major_color_kmeans(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    color = img.reshape((-1, 3))
    color = np.float32(color)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    vectors, labels, centers = cv2.kmeans(color, 3, None, criteria, 10, flags)

    centers = np.int32(centers)
    print(centers)
    return centers


def show_color(colours, fig):
    color_3 = []
    for color in colours:
        color_temp = [color[1][0] / 255, color[1][1] / 255, color[1][2] / 255]
        color_3.append(color_temp)
    fig.axis([0, 3, 0, 1])
    fig.bar([0.5, 1.5, 2.5], 1, width=1, color=list(color_3))


def show_color_hsv(colour, fig):
    count = 0
    fig.axis([0, 3, 0, 2])
    for center in colour:
        r, g, b = colorsys.hsv_to_rgb(center[0] / 180, center[1] / 255, center[2] / 255)
        color = [r, g, b]
        print("rbg", [r * 255, g * 255, b * 255])
        fig.add_patch(patches.Rectangle((count, 0), 1, 1, color=color))
        # color = [center[0]/255, center[1]/255, center[2]/255]
        if center[1] < 80 and 160 > center[2] > 80:
            print("Gray")
        elif center[1] < 80 and center[2] < 80:
            print("Special")
        elif center[2] < 80:
            center[2] = 10
            print("Black")
        elif center[1] < 80:
            center[1] = 10
            print("White")
        if 180 > center[1] >= 80:
            center[1] *= 1.2
        if 180 > center[2] >= 80:
            center[2] *= 1.2
        if 160 > center[0] > 30:
            print("Invalid")
            center[0] = 15
        print("h:", center[0], "s:", center[1], "v:", center[2])
        r, g, b = colorsys.hsv_to_rgb(center[0] / 180, center[1] / 255, center[2] / 255)
        color = [r, g, b]
        fig.add_patch(patches.Rectangle((count, 1), 1, 1, color=color))
        count += 1


def major_color_palette(img, num_colors):
    colours = img.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
    colours = colours.convert('RGB')
    major_colors = colours.getcolors()
    print(major_colors)
    return major_colors


