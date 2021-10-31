import numpy as np
import onnx
import onnxruntime as ort
import cv2
import os
import cv_process as cv
from PIL import Image
import matplotlib.pyplot as plt


def main(img):
    im_height = 224
    im_width = 224

    # labels = ["Bicolor", "Calico", "Colorpoint", "Mix", "Orange", "Solid", "Tabby"]

    img = img.resize((im_width, im_height))
    img = np.array(img).astype(np.float32)

    ft = img.flatten()
    for i in range(0, 10):
        print(ft[i * 3 + 0], ",", ft[i * 3 + 1], ",", ft[i * 3 + 2])
    print("///")
    for i in range(224, 234):
        print(ft[i * 3 + 0], ",", ft[i * 3 + 1], ",", ft[i * 3 + 2])
    print("///")
    for i in range(448, 458):
        print(ft[i * 3 + 0], ",", ft[i * 3 + 1], ",", ft[i * 3 + 2])
    print("///")
    for i in range(672, 682):
        print(ft[i * 3 + 0], ",", ft[i * 3 + 1], ",", ft[i * 3 + 2])

    img = ((img / 255.) - 0.5) * 2.0
    img = (np.expand_dims(img, 0))



    onnx_model = onnx.load_model("model11v6.onnx")
    sess = ort.InferenceSession(onnx_model.SerializeToString())
    sess.set_providers(['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output = sess.run([output_name], {input_name: img})[0]

    prob = np.squeeze(output[0])
    print("predicting label:", np.argmax(prob), "\nprob:", max(prob))


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
        plt_img = Image.open(img_path)
        cat = Image.fromarray(cat)
        full_cat = cv.recolor(full_cat)
        # show and test
        plt.subplot(211)
        plt.imshow(plt_img)
        """
        print("Coordinates:",
              coordinate[0], coordinate[1], coordinate[2], coordinate[3])
        print("Size:", coordinate[4], "*", coordinate[4])
        """
        main(plt_img)
        fig = plt.subplot(212)
        cv.show_color_hsv(colours, fig)
        plt.show()
    else:
        print("Cannot detect cat!!!")








