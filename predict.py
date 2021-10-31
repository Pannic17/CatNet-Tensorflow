import os
import json
import glob
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import cv_process as cv
import cv2

from model_v2 import MobileNetV2


def main(img):
    im_height = 224
    im_width = 224
    num_classes = 7

    labels = {
        "0": "Bicolor",
        "1": "Calico",
        "2": "Colorpoint",
        "3": "Mix",
        "4": "Orange",
        "5": "Solid",
        "6": "Tabby"
        }

    # load image
    # img_path = "../tulip.jpg"
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path)
    # resize image to 224x224
    img = img.resize((im_width, im_height))
    # plt.imshow(img)

    # scaling pixel value to (-1,1)
    img = np.array(img).astype(np.float32)
    img = ((img / 255.) - 0.5) * 2.0

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    feature = MobileNetV2(include_top=False)
    model = tf.keras.Sequential([feature,
                                 tf.keras.layers.GlobalAvgPool2D(),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(num_classes),
                                 tf.keras.layers.Softmax()])
    weights_path = './save_weights/CMN_b16e24_onnx_v5.ckpt'
    assert len(glob.glob(weights_path+"*")), "cannot find {}".format(weights_path)
    model.load_weights(weights_path)

    result = np.squeeze(model.predict(img))
    predict_class = np.argmax(result)

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
                                                 result[predict_class])
    plt.title(print_res)
    print(print_res)


if __name__ == '__main__':
    file_name = input("Filename: ")
    img_path = "./test/" + file_name + ".jpg"
    assert os.path.exists(img_path), "File: '{}' dose not exist.".format(img_path)
    # get cat face and coordinate
    cat, coordinate = cv.cut(img_path)
    if cat is not None:
        # modify the image via opencv
        cat = cv.recolor(cat)
        full_cat = cv2.imread(img_path)
        full_cat = cv2.rectangle(full_cat,
                                 (coordinate[0], coordinate[1]),
                                 (coordinate[2], coordinate[3]),
                                 (0, 0, 255), 2)
        # transform the image to PIL image type
        colours = cv.major_color_kmeans(cat)
        cat = Image.fromarray(cat)
        full_cat = cv.recolor(full_cat)
        # show and test
        plt.subplot(211)
        plt.imshow(full_cat)
        print("Coordinates:",
              coordinate[0], coordinate[1], coordinate[2], coordinate[3])
        print("Size:", coordinate[4], "*", coordinate[4])
        main(cat)
        fig = plt.subplot(212)
        cv.show_color_hsv(colours, fig)
        plt.show()
    else:
        print("Cannot detect cat!!!")
