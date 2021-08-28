import cv2
import numpy as np
from PIL import Image
import logging

logging.getLogger("tensorflow").setLevel(logging.WARNING)
import tensorflow as tf
from tensorflow.keras.models import load_model


def load_classifier_model(weights_path: str):
    """
    Loads Tensorflow classifier model

    Args:
        weights_path (str): path to weights file

    Returns:
        classifier_model: tensorflow model object
    """
    classifier_model = load_model(weights_path)

    return classifier_model


def convert_image_arr_to_tensor(image_array):
    """
    Converts image numpy array to a tensor object and also resizes it as well as expands it dimension to feed to the model

    Args:
    image_array: numpy array

    Returns:
    final_tensor: tensor
    """

    tensor = tf.convert_to_tensor(image_array)

    resized_img = tf.image.resize(tensor, [512, 512], method="nearest")

    final_tensor = tf.expand_dims(resized_img, axis=0)

    final_tensor = tf.cast(final_tensor, tf.float32) / 255.0

    return final_tensor


def classify_image(tensor_model, img_tensor):
    """
    Uses the classifier model to classify the input as belonging to one of the categories

    Args:
    tensor_model: tensor model object, img_tensor: tensor object of the image

    Returns:
    class_name: Either "Covid" or "Non_Covid"
    Note: The last layer of the model uses a sigmoid function to classify the images. Now, a sigmoid function values fall in the gamut of 0-1.
    Now, we set our threshold as 0.5 with values less than 0.5 being classfies as "Covid" and more than 0.5 as "Non_Covid"
    """
    class_dict = {0: "Covid", 1: "Non_Covid"}

    predictions = tensor_model.predict(img_tensor)

    if predictions[0][0] < 0.5:
        class_name = class_dict[0]

    else:
        class_name = class_dict[1]

    return class_name
