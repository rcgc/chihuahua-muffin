import matplotlib.pyplot as plt
import os, shutil, pathlib
import numpy as np
from tensorflow.keras.preprocessing import image


def predict_image_class(category, id):
    img_path = f"./{category}/{category}_{id}.JPG"
    # img_path = "./readme_images/camouflaged_owl.jpg"
    plt.figure()
    image_matrix = plt.imread(img_path)
    plt.imshow(image_matrix)
    plt.axis("off")

    img = image.load_img(img_path, target_size=(112, 112))
    img_array = image.img_to_array(img)
    x = np.expand_dims(img_array, axis=0)

    img_preprocessed = np.vstack([x])

    return img_preprocessed
