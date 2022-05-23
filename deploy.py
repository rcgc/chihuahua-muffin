import matplotlib.pyplot as plt
import os, shutil, pathlib
import numpy as np
from tensorflow.keras.preprocessing import image


def predict_image_class(model, class_labels_map, category, id):
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

    predictions = model.predict(img_preprocessed)
    print("\nProbabilities : ", predictions)

    max_index_probability = np.argmax(predictions)
    print("\nPredicted : ", list(class_labels_map.keys())[list(class_labels_map.values()).index(max_index_probability)])

    plt.show(block=False)
